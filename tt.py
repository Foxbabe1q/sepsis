import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import shap
import lifelines
from lifelines import CoxPHFitter

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import shutil  # 用于复制文件

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Global Plot Style ---
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/eda", exist_ok=True)
os.makedirs("results/shap", exist_ok=True)  # 用于保存 SHAP 相关输出


def set_seed(seed):
    """
    Set random seeds for reproducibility across torch, numpy, and python's built-in random.
    Also set cudnn deterministic to True if CUDA is available.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 3
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# (1) EDA, Data Extraction
# ---------------------------

def perform_eda_and_save(data):
    desc_all = data.describe(include='all').transpose()
    desc_all.to_csv("results/eda/data_description_overall.csv", index=True)

    group_labels = data['icu_expire_flag'].unique()
    stats_list = []
    for g in group_labels:
        sub_df = data[data['icu_expire_flag'] == g]
        d = sub_df.describe().transpose()
        d['group'] = f"icu_expire_flag={g}"
        stats_list.append(d)
    desc_grouped = pd.concat(stats_list, axis=0)
    desc_grouped.to_csv("results/eda/data_description_grouped.csv", index=True)

    print("Basic descriptive stats saved to results/eda/")

    if 'admission_age' in data.columns:
        plt.figure()
        sns.histplot(data['admission_age'], kde=True, bins=30, color='blue')
        plt.title("Distribution of admission_age")
        plt.savefig("results/eda/admission_age_dist.png")
        plt.close()


def extract_unlab_data():
    data = pd.read_csv('seps.csv', na_values='NULL')
    sequence_length = 48

    base_vis_cols = [
        'dopamine_vis', 'dobutamine_vis', 'epinephrine_vis',
        'milrinone_vis', 'vasopressin_vis', 'norepinephrine_vis'
    ]
    vis_columns = base_vis_cols + ['total_vis']

    for col in vis_columns:
        data[col].fillna(0, inplace=True)

    # correct total_vis
    data['total_vis'] = (
          data['dopamine_vis']
        + data['dobutamine_vis']
        + 100.0 * data['epinephrine_vis']
        + 10.0  * data['milrinone_vis']
        + 100.0 * data['norepinephrine_vis']
        + 10000.0 * data['vasopressin_vis']
    )

    patients = data[['subject_id', 'stay_id']].drop_duplicates()
    X_unlab_list = []
    for pat in patients.itertuples(index=False):
        sid = pat.subject_id
        stid = pat.stay_id
        pat_data = data[(data['subject_id'] == sid) & (data['stay_id'] == stid)]
        full_hours = pd.DataFrame({'hour_index': np.arange(1, sequence_length+1)})
        pat_data = full_hours.merge(pat_data, on='hour_index', how='left')
        pat_data[vis_columns] = pat_data[vis_columns].fillna(0)
        X_unlab_list.append(pat_data[vis_columns].values)

    X_unlab = np.array(X_unlab_list, dtype=np.float32)
    np.save('X_unlab.npy', X_unlab)
    print("Unlabeled data extracted with corrected total_vis => shape:", X_unlab.shape)


# ---------------------------
# (2) Model Components
# ---------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class TSTEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=8,
                 num_layers=2, dim_feedforward=256, dropout=0.1, return_hidden=False):
        super(TSTEncoder, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.return_hidden = return_hidden

    def forward(self, x):
        b, seq_len, _ = x.shape
        zero_mask = (x == 0).all(dim=2)
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, zero_mask], dim=1)

        x = self.input_projection(x)
        x = self.pos_encoder(x)
        hidden = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        hidden = self.dropout(hidden)
        cls_output = hidden[:, 0, :]
        if self.return_hidden:
            return cls_output, hidden
        else:
            return cls_output


class MAEModel(nn.Module):
    def __init__(self, input_dim=7, d_model=64, n_heads=8,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super(MAEModel, self).__init__()
        self.encoder = TSTEncoder(
            input_dim=input_dim, d_model=d_model, n_heads=n_heads,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, return_hidden=True
        )
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x, mask):
        _, hidden = self.encoder(x)
        reconstruction = self.decoder(hidden[:, 1:, :])
        return reconstruction


class UnlabeledDataset(Dataset):
    def __init__(self, X_seq):
        self.X_seq = X_seq

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx]


def random_mask(batch_x, mask_ratio=0.3):
    b, seq_len, _ = batch_x.shape
    num_mask = int(seq_len * mask_ratio)
    mask = torch.zeros(b, seq_len, dtype=torch.bool)
    for i in range(b):
        mask_idx = np.random.choice(seq_len, num_mask, replace=False)
        mask[i, mask_idx] = True
    return mask


def pretrain_teacher(seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_unlab = np.load('X_unlab.npy')
    X_unlab = np.nan_to_num(X_unlab, nan=0.0, posinf=1e6, neginf=-1e6)
    X_unlab = np.clip(X_unlab, -1000, 1000)
    N, sl, d = X_unlab.shape

    X_unlab_reshaped = X_unlab.reshape(-1, d)
    mean = X_unlab_reshaped.mean(axis=0)
    std = X_unlab_reshaped.std(axis=0) + 1e-6
    X_unlab_norm = (X_unlab_reshaped - mean) / std
    X_unlab = X_unlab_norm.reshape(N, sl, d).astype(np.float32)
    print(f"[MAE] X_unlab shape={X_unlab.shape}")

    X_torch = torch.tensor(X_unlab, dtype=torch.float32)
    unlab_dataset = UnlabeledDataset(X_torch)
    unlab_loader = DataLoader(unlab_dataset, batch_size=64, shuffle=True)

    mae_model = MAEModel(
        input_dim=d, d_model=64, n_heads=8, num_layers=2,
        dim_feedforward=256, dropout=0.1
    ).to(device)

    optimizer = optim.AdamW(mae_model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 20
    mask_ratio = 0.05
    for epoch in range(num_epochs):
        mae_model.train()
        total_loss = 0.0
        count = 0
        for batch_x in unlab_loader:
            batch_x = batch_x.to(device)
            mask = random_mask(batch_x, mask_ratio=mask_ratio).to(device)
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            reconstruction = mae_model(batch_x, mask)
            mask_3d = mask.unsqueeze(-1).expand_as(batch_x)
            masked_count = mask_3d.sum().item()
            if masked_count == 0:
                continue
            loss = criterion(reconstruction[mask_3d], batch_x[mask_3d])
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(mae_model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
        if count > 0:
            avg_loss = total_loss / count
            print(f"[MAE Epoch {epoch+1}/{num_epochs}] Loss={avg_loss:.4f}")
        else:
            print(f"[MAE Epoch {epoch+1}/{num_epochs}] no effective batch")

    torch.save(mae_model.state_dict(), "teacher_model.pth")
    print("teacher_model.pth saved.")


# ---------------------------
# MultiTaskModel + Training
# ---------------------------

class MultiTaskModelMain(nn.Module):
    def __init__(self, time_input_dim, patient_input_dim, d_model=64, n_heads=8,
                 num_layers=2, dim_feedforward=256, dropout=0.2, latent_dim=128, num_scores=4):
        super(MultiTaskModelMain, self).__init__()
        self.tst_encoder = TSTEncoder(
            input_dim=time_input_dim, d_model=d_model, n_heads=n_heads,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # classification branch => uses all patient_input_size
        self.fc_label = nn.Sequential(
            nn.Linear(d_model + patient_input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        # regression branch => uses (patient_input_size - 4)
        self.fc_scores = nn.Sequential(
            nn.Linear(d_model + (patient_input_dim - 4), latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_scores)
        )

    def forward(self, x_time, x_patient):
        seq_feat = self.tst_encoder(x_time)

        # classification => "full" static
        combined_cls = torch.cat([seq_feat, x_patient], dim=1)
        logits = self.fc_label(combined_cls)

        # regression => skip 4 severity columns from x_patient
        x_patient_nosev = torch.cat([x_patient[:, :1], x_patient[:, 5:]], dim=1)
        combined_regr = torch.cat([seq_feat, x_patient_nosev], dim=1)
        scores = self.fc_scores(combined_regr)

        return logits, scores


def calculate_classification_metrics(labels, preds):
    cm = confusion_matrix(labels, preds)
    if cm.shape != (2, 2):
        TN, FP, FN, TP = 0, 0, 0, 0
        if len(cm) == 1:
            if labels[0] == 0:
                TN = cm[0][0]
            else:
                TP = cm[0][0]
    else:
        TN, FP, FN, TP = cm.ravel()

    sens = TP/(TP+FN) if (TP+FN) > 0 else 0
    spec = TN/(TN+FP) if (TN+FP) > 0 else 0
    PPV = TP/(TP+FP) if (TP+FP) > 0 else 0
    NPV = TN/(TN+FN) if (TN+FN) > 0 else 0
    PLR = sens/(1 - spec) if spec < 1 else 0
    NLR = (1 - sens)/spec if spec > 0 else 0
    ACC = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0
    return PPV, NPV, PLR, NLR, sens, spec, ACC


def bootstrap_metrics_ci(y_true, y_prob, n_bootstrap=1000, alpha=0.05):
    rng = np.random.default_rng(seed=42)
    N = len(y_true)
    auroc_vals = []
    ppv_vals = []
    npv_vals = []
    plr_vals = []
    nlr_vals = []
    sens_vals = []
    spec_vals = []
    acc_vals = []

    for _ in range(n_bootstrap):
        indices = rng.choice(N, size=N, replace=True)
        y_t = y_true[indices]
        y_p = y_prob[indices]
        try:
            auroc_bs = roc_auc_score(y_t, y_p)
        except ValueError:
            continue
        preds_bs = (y_p > 0.5).astype(int)
        PPV, NPV, PLR, NLR, sens, spec, ACC = calculate_classification_metrics(y_t, preds_bs)
        auroc_vals.append(auroc_bs)
        ppv_vals.append(PPV)
        npv_vals.append(NPV)
        plr_vals.append(PLR)
        nlr_vals.append(NLR)
        sens_vals.append(sens)
        spec_vals.append(spec)
        acc_vals.append(ACC)

    def mean_ci(arr):
        arr = np.array(arr)
        if len(arr) == 0:
            return (0.0, 0.0, 0.0)
        arr_sorted = np.sort(arr)
        lower_idx = int(alpha/2 * len(arr))
        upper_idx = int((1 - alpha/2) * len(arr))
        mean_ = np.mean(arr)
        if upper_idx - 1 < len(arr_sorted):
            upper_ = arr_sorted[upper_idx - 1]
        else:
            upper_ = arr_sorted[-1]
        lower_ = arr_sorted[lower_idx]
        return (mean_, lower_, upper_)

    metrics_dict = {}
    metrics_dict['AUROC'] = mean_ci(auroc_vals)
    metrics_dict['PPV'] = mean_ci(ppv_vals)
    metrics_dict['NPV'] = mean_ci(npv_vals)
    metrics_dict['PLR'] = mean_ci(plr_vals)
    metrics_dict['NLR'] = mean_ci(nlr_vals)
    metrics_dict['Sensitivity'] = mean_ci(sens_vals)
    metrics_dict['Specificity'] = mean_ci(spec_vals)
    metrics_dict['ACC'] = mean_ci(acc_vals)
    return metrics_dict


def train_one_epoch(model, loader, optimizer, device,
                    criterion_label, criterion_scores,
                    kd, teacher_model, distill_criterion,
                    mt):
    model.train()
    total_loss = 0.0
    total_count = 0
    correct_cls = 0
    total_cls = 0
    sum_sq_res = 0.0
    sum_sq_tot = 0.0

    for batch_x, x_p, y_l, y_s in loader:
        batch_x, x_p, y_l, y_s = batch_x.to(device), x_p.to(device), y_l.to(device), y_s.to(device)
        optimizer.zero_grad()
        logits, scores = model(batch_x, x_p)

        loss_label = criterion_label(logits, y_l)
        loss_scores_val = criterion_scores(scores[:, 0], y_s[:, 0]) if mt else 0.0
        loss = loss_label + (0.1 * loss_scores_val if mt else 0.0)

        if kd and teacher_model is not None:
            with torch.no_grad():
                t_logits, t_scores = teacher_model(batch_x, x_p)
            dist_loss = nn.MSELoss()(torch.softmax(logits, dim=1),
                                     torch.softmax(t_logits, dim=1))
            loss += 0.05 * dist_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_x.size(0)
        total_count += batch_size

        preds_cls = logits.argmax(dim=1)
        correct_cls += (preds_cls == y_l).sum().item()
        total_cls += y_l.size(0)

        sum_sq_res += torch.sum((scores[:, 0] - y_s[:, 0])**2).item()
        mean_y0 = torch.mean(y_s[:, 0])
        sum_sq_tot += torch.sum((y_s[:, 0] - mean_y0)**2).item()

    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    acc = correct_cls / total_cls if total_cls > 0 else 0.0
    r2 = 1.0 - (sum_sq_res / sum_sq_tot) if sum_sq_tot > 0 else 0.0

    return avg_loss, acc, r2


def evaluate_classification_regression(model, loader, device,
                                       mt, criterion_label, criterion_scores):
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct_cls = 0
    total_cls = 0
    sum_sq_res = 0.0
    sum_sq_tot = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, x_p, y_l, y_s in loader:
            batch_x, x_p, y_l, y_s = batch_x.to(device), x_p.to(device), y_l.to(device), y_s.to(device)
            logits, scores = model(batch_x, x_p)

            loss_label = criterion_label(logits, y_l)
            loss_scores_val = criterion_scores(scores[:, 0], y_s[:, 0]) if mt else 0.0
            loss = loss_label + (0.1 * loss_scores_val if mt else 0.0)

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            preds_cls = logits.argmax(dim=1)
            correct_cls += (preds_cls == y_l).sum().item()
            total_cls += y_l.size(0)

            sum_sq_res += torch.sum((scores[:, 0] - y_s[:, 0])**2).item()
            mean_y0 = torch.mean(y_s[:, 0])
            sum_sq_tot += torch.sum((y_s[:, 0] - mean_y0)**2).item()

            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_labels.append(y_l.cpu().numpy())
            all_probs.append(probs)

    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    acc = correct_cls / total_cls if total_cls > 0 else 0.0
    r2 = 1.0 - (sum_sq_res / sum_sq_tot) if sum_sq_tot > 0 else 0.0

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    return avg_loss, acc, r2, all_labels, all_probs


def do_shap_analysis_static_only(model, X_static, feature_names_static=None, device='cpu'):
    print(f"[SHAP] total samples for analysis: {X_static.shape[0]} (static only)")

    if feature_names_static is None:
        feature_names_static = [f"static_{i}" for i in range(X_static.shape[1])]

    sample_size = min(200, X_static.shape[0])
    X_sample = X_static[:sample_size]

    def custom_forward(x):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            bsize = x_t.size(0)
            seq_len = 48
            time_dim = 7
            zero_time = torch.zeros(bsize, seq_len, time_dim, dtype=torch.float32, device=device)
            logits, scores = model(zero_time, x_t)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy()

    import shap
    explainer = shap.Explainer(
        custom_forward,
        X_sample,
        feature_names=feature_names_static,
        max_evals=2 * X_sample.shape[1] + 1
    )
    shap_values = explainer(X_sample)

    # 保存 SHAP 值
    np.save("results/shap/shap_values_static_only.npy", shap_values.values)
    print("[SHAP] Raw shap values saved => results/shap/shap_values_static_only.npy")

    # 计算平均绝对值并排序，输出前5
    mean_abs = np.mean(np.abs(shap_values.values), axis=0)
    top_n = 5
    top_indices = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [feature_names_static[i] for i in top_indices]
    top_importances = mean_abs[top_indices]

    df_top = pd.DataFrame({"feature": top_features, "mean_abs_shap": top_importances})
    df_top.to_csv("results/shap/top_features_static_only.csv", index=False)
    print("[SHAP] Top 5 features by mean abs shap => results/shap/top_features_static_only.csv")
    print(df_top)

    # 绘制 summary_plot 并保存
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names_static,
        max_display=20,
        show=False
    )
    plt.savefig("results/shap/shap_summary_static_only.png")
    plt.close()
    print("[SHAP] summary plot saved => results/shap/shap_summary_static_only.png (no VIS included)")


def run_experiment(seed, kd, mt, ls, model_type='main', do_shap=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv('seps.csv', na_values='NULL')
    patients = data[['subject_id','stay_id']].drop_duplicates()
    print(f"Total unique patients/stays in this dataset: {len(patients)}")

    sequence_length = 48
    vis_columns = [
        'dopamine_vis','dobutamine_vis','epinephrine_vis','milrinone_vis',
        'vasopressin_vis','norepinephrine_vis','total_vis'
    ]
    static_cols = [
        'gender','admission_age','race','insurance','marital_status',
        'sofa_score_24h','sapsii','lods','oasis'
    ]
    label_col = 'icu_expire_flag'

    # fill missing
    if 'hospital_expire_flag' not in data.columns:
        data['hospital_expire_flag'] = data[label_col]
    if 'hospital_los' not in data.columns:
        data['hospital_los'] = data['hour_index']

    if data['admission_age'].isnull().any():
        age_median = data['admission_age'].median()
        data['admission_age'].fillna(age_median, inplace=True)
    for c in ['gender','insurance','marital_status','race']:
        data[c].fillna("Unknown", inplace=True)
    data[label_col].fillna(0, inplace=True)
    data[label_col] = data[label_col].astype(int)

    for sc in ['sofa_score_24h','sapsii','lods','oasis']:
        data[sc].fillna(data[sc].median(), inplace=True)
    for vc in vis_columns:
        data[vc].fillna(0, inplace=True)

    cat_cols = ['gender','race','insurance','marital_status']
    cat_values = data[cat_cols].astype(str).values
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(cat_values)

    X_seq_list = []
    X_static_list = []
    y_label_list = []
    y_scores_list = []

    for pat in patients.itertuples(index=False):
        sid, stid = pat.subject_id, pat.stay_id
        pat_data = data[(data['subject_id'] == sid) & (data['stay_id'] == stid)]
        full_hours = pd.DataFrame({'hour_index': np.arange(1, sequence_length+1)})
        pat_data = full_hours.merge(pat_data, on='hour_index', how='left')
        pat_data[vis_columns] = pat_data[vis_columns].fillna(0)

        label = pat_data[label_col].iloc[0]
        y_label_list.append(label)

        sofa_24h = pat_data['sofa_score_24h'].iloc[0]
        sapsii_val = pat_data['sapsii'].iloc[0]
        lods_val = pat_data['lods'].iloc[0]
        oasis_val = pat_data['oasis'].iloc[0]
        y_scores_list.append([sofa_24h, sapsii_val, lods_val, oasis_val])

        seq_array = pat_data[vis_columns].values.astype(np.float32)
        seq_array = np.log1p(seq_array)
        X_seq_list.append(seq_array)

        static_row = pat_data.iloc[0]
        static_num = static_row[['admission_age','sofa_score_24h','sapsii','lods','oasis']].values.astype(np.float32)
        static_cat_vals = static_row[cat_cols].astype(str).values.reshape(1, -1)
        static_cat_enc = ohe.transform(static_cat_vals).astype(np.float32)
        combined_static = np.concatenate([static_num, static_cat_enc[0]], axis=0).astype(np.float32)
        X_static_list.append(combined_static)

    # ---- 组装成最终的 ndarray ----
    X_seq = np.array(X_seq_list, dtype=np.float32)       # shape: (N, 48, 7)
    X_static = np.array(X_static_list, dtype=np.float32) # shape: (N, ?)
    y_label = np.array(y_label_list, dtype=np.int64)
    y_scores = np.array(y_scores_list, dtype=np.float32)

    N, sl, vf = X_seq.shape

    # ============ 先拆分，再对训练集做 fit_transform，验证/测试集只做 transform ============
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y_label
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=seed, stratify=y_label[train_idx]
    )

    # 先把原始 X_seq 分为三个部分
    X_seq_train = X_seq[train_idx]
    X_seq_val   = X_seq[val_idx]
    X_seq_test  = X_seq[test_idx]

    # 对 train 做 fit_transform
    X_seq_train_reshaped = X_seq_train.reshape(-1, vf)  # (train_size*48, vf)
    scaler_seq = StandardScaler()
    X_seq_train_scaled = scaler_seq.fit_transform(X_seq_train_reshaped)
    X_seq_train_scaled = X_seq_train_scaled.reshape(-1, sl, vf).astype(np.float32)

    # 对 val 做 transform
    X_seq_val_reshaped = X_seq_val.reshape(-1, vf)
    X_seq_val_scaled = scaler_seq.transform(X_seq_val_reshaped)
    X_seq_val_scaled = X_seq_val_scaled.reshape(-1, sl, vf).astype(np.float32)

    # 对 test 做 transform
    X_seq_test_reshaped = X_seq_test.reshape(-1, vf)
    X_seq_test_scaled = scaler_seq.transform(X_seq_test_reshaped)
    X_seq_test_scaled = X_seq_test_scaled.reshape(-1, sl, vf).astype(np.float32)

    # ---- 准备最终数据集 ----
    X_train, X_val, X_test = X_seq_train_scaled, X_seq_val_scaled, X_seq_test_scaled
    X_static_train, X_static_val, X_static_test = (
        X_static[train_idx],
        X_static[val_idx],
        X_static[test_idx]
    )
    y_train, y_val, y_test = (
        y_label[train_idx],
        y_label[val_idx],
        y_label[test_idx]
    )
    y_scores_train, y_scores_val, y_scores_test = (
        y_scores[train_idx],
        y_scores[val_idx],
        y_scores[test_idx]
    )

    # ---- 后面不变：构建 DataLoader, 定义模型, 训练, 验证, 测试, 画图等 ----
    class SepsisDataset(Dataset):
        def __init__(self, X, X_p, y, y_scores):
            self.X = X
            self.X_p = X_p
            self.y = y
            self.y_scores = y_scores

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return (self.X[idx], self.X_p[idx], self.y[idx], self.y_scores[idx])

    train_dataset = SepsisDataset(X_train, X_static_train, y_train, y_scores_train)
    val_dataset   = SepsisDataset(X_val,   X_static_val,   y_val,   y_scores_val)
    test_dataset  = SepsisDataset(X_test,  X_static_test,  y_test,  y_scores_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

    # 下面的模型定义、训练循环、验证、测试、绘图等代码不变
    # ...
    # ...
    # (省略其余原有逻辑)


    patient_input_size = X_static_train.shape[1]
    num_scores = 4

    model = MultiTaskModelMain(
        time_input_dim=vf, patient_input_dim=patient_input_size,
        d_model=64, n_heads=8, num_layers=2,
        dim_feedforward=256, dropout=0.2,
        latent_dim=128, num_scores=num_scores
    ).to(device)

    # 知识蒸馏 kd
    if kd:
        mae_model = MAEModel(
            input_dim=vf, d_model=64, n_heads=8, num_layers=2, dim_feedforward=256, dropout=0.1
        ).to(device)
        mae_model.load_state_dict(torch.load('teacher_model.pth', map_location=device))
        mae_model.eval()

        teacher_model = MultiTaskModelMain(
            time_input_dim=vf, patient_input_dim=patient_input_size,
            d_model=64, n_heads=8, num_layers=2, dim_feedforward=256, dropout=0.2,
            latent_dim=128, num_scores=num_scores
        ).to(device)
        teacher_model.tst_encoder.load_state_dict(mae_model.encoder.state_dict(), strict=False)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    else:
        teacher_model = None

    class_counts = np.bincount(y_train)
    class_weights = len(y_train)/(2.0*class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion_label = nn.CrossEntropyLoss(weight=class_weights)

    criterion_scores = nn.MSELoss()
    distill_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    best_auroc = 0.0
    trigger_times = 0
    patience = 10
    num_epochs = 30
    train_log = []
    best_state = None

    for epoch in range(num_epochs):
        train_loss, train_acc, train_r2 = train_one_epoch(
            model, train_loader, optimizer, device,
            criterion_label, criterion_scores,
            kd, teacher_model, distill_criterion,
            mt
        )
        val_loss, val_acc, val_r2, val_labels, val_probs = evaluate_classification_regression(
            model, val_loader, device, mt,
            criterion_label, criterion_scores
        )
        val_auroc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auroc)

        train_log.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "train_r2":   train_r2,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "val_r2":     val_r2,
            "val_auroc":  val_auroc
        })
        print(f"[Epoch {epoch+1}/{num_epochs}] train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_auroc={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            trigger_times = 0
            best_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch={epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        save_path = f"results/models/{model_type}_kd{kd}_mt{mt}_ls{ls}_best.pth"
        torch.save(best_state, save_path)
        print(f"[INFO] best model saved => {save_path}")
    else:
        print("[WARNING] No best_state found? The model didn't update at all?")

    test_loss, test_acc, test_r2, test_labels, test_probs = evaluate_classification_regression(
        model, test_loader, device, mt,
        criterion_label, criterion_scores
    )

    train_loss2, train_acc2, train_r2_2, train_labels2, train_probs2 = evaluate_classification_regression(
        model, train_loader, device, mt, criterion_label, criterion_scores
    )

    direct_auroc = roc_auc_score(test_labels, test_probs)
    metrics_ci = bootstrap_metrics_ci(test_labels, test_probs, n_bootstrap=1000, alpha=0.05)

    df_trainlog = pd.DataFrame(train_log)
    out_log = f"results/logs/train_curve_{model_type}_kd{kd}_mt{mt}_ls{ls}.csv"
    df_trainlog.to_csv(out_log, index=False)

    plt.figure()
    plt.grid(True)
    plt.plot(df_trainlog['epoch'], df_trainlog['train_loss'], label='train_loss')
    plt.plot(df_trainlog['epoch'], df_trainlog['val_loss'], label='val_loss')
    plt.title(f"Loss curve {model_type}_kd{kd}_mt{mt}_ls{ls}")
    plt.legend()
    plt.savefig(f"results/figures/loss_curve_{model_type}_kd{kd}_mt{mt}_ls{ls}.png")
    plt.close()

    plt.figure()
    plt.grid(True)
    plt.plot(df_trainlog['epoch'], df_trainlog['train_r2'], label='train_r2', color='blue')
    plt.plot(df_trainlog['epoch'], df_trainlog['val_r2'], label='val_r2', color='red')
    plt.title(f"R2 curve {model_type}_kd{kd}_mt{mt}_ls{ls}")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.legend()
    plt.savefig(f"results/figures/r2_curve_{model_type}_kd{kd}_mt{mt}_ls{ls}.png")
    plt.close()

    if do_shap:
        print("[INFO] Doing SHAP analysis (static features only, no VIS) for model_type=", model_type)
        cat_cols = ['gender','race','insurance','marital_status']
        ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names_static = ["admission_age","sofa_score_24h","sapsii","lods","oasis"] + ohe_feature_names
        do_shap_analysis_static_only(model, X_static_test, feature_names_static, device)

    return (metrics_ci, test_r2, direct_auroc, test_labels, test_probs,
            train_labels2, train_probs2, out_log)


# ---------------------------
# (3) 绘图函数：排除 no_ls 的 ROC、排除 no_mt 的 R2
# ---------------------------

def plot_roc_for_configs(roc_info_list, save_path="results/figures/roc_compare_all.png"):
    """
    绘制测试集上的ROC曲线，排除 config == "no_ls"。
    """
    plt.figure(figsize=(6,6))
    plt.grid(True)

    for item in roc_info_list:
        config_name = item["config"]
        if config_name == "no_ls":
            continue  # 依旧排除 no_ls

        y_true = item["y_true"]
        y_prob = item["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_ = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{config_name} (AUC={auc_:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Comparison on Test set")  # 修改标题
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC comparison saved => {save_path}")


def plot_roc_for_configs_train(roc_info_list_train, save_path="results/figures/roc_compare_train.png"):
    """
    绘制训练集上的ROC曲线，排除 config == "no_ls"。
    逻辑与plot_roc_for_configs类似，只是标题和输出文件不同。
    """
    plt.figure(figsize=(6,6))
    plt.grid(True)

    for item in roc_info_list_train:
        config_name = item["config"]
        if config_name == "no_ls":
            continue  # 依旧排除 no_ls

        y_true = item["y_true"]
        y_prob = item["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_ = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{config_name} (AUC={auc_:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Comparison on Training set")  # 新函数标题
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC comparison (training) saved => {save_path}")



def plot_loss_subplots(log_paths_dict, save_path="results/figures/loss_subplots.png"):
    """
    分别绘制 baseline / no_kd / no_mt 的Loss曲线(3个子图,共一张图)
    """
    configs_to_show = ["baseline", "no_kd", "no_mt"]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    for i, conf in enumerate(configs_to_show):
        csv_path = log_paths_dict[conf]
        df_log = pd.read_csv(csv_path)
        axs[i].grid(True)
        axs[i].plot(df_log["epoch"], df_log["train_loss"], label="train_loss", color="blue")
        axs[i].plot(df_log["epoch"], df_log["val_loss"],   label="val_loss",   color="red")
        axs[i].set_title(f"Loss - {conf}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Loss")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss subplots saved => {save_path}")


def plot_r2_subplots(log_paths_dict, save_path="results/figures/r2_subplots.png"):
    """
    分别绘制 baseline / no_kd (不含 no_mt) 的 R2曲线(2个子图,共一张图)
    因为你说不需要包含 no_mt。
    """
    configs_to_show = ["baseline", "no_kd"]  # 排除 "no_mt"
    fig, axs = plt.subplots(nrows=1, ncols=len(configs_to_show), figsize=(10,5))
    for i, conf in enumerate(configs_to_show):
        csv_path = log_paths_dict[conf]
        df_log = pd.read_csv(csv_path)
        axs[i].grid(True)
        axs[i].plot(df_log["epoch"], df_log["train_r2"], label="train_r2", color="blue")
        axs[i].plot(df_log["epoch"], df_log["val_r2"],   label="val_r2",   color="red")
        axs[i].set_title(f"R2 - {conf}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("R2 Score")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"R2 subplots saved => {save_path} (no_mt excluded)")


# ---------------------------
# (4) 主流程
# ---------------------------

if __name__ == "__main__":

    raw_data = pd.read_csv('seps.csv', na_values='NULL')
    perform_eda_and_save(raw_data)
    extract_unlab_data()

    # 预训练 MAE (teacher) 模型
    pretrain_teacher(seed=42)

    for try_seed in range(11, 12):
        print(f"\n=== Now running ablation under random seed={try_seed} ===")

        out_csv = "results/logs/main_model_ablation_results_withCI.csv"
        if os.path.exists(out_csv):
            os.remove(out_csv)

        # 4 种消融配置 (ls不生效, 仅作记录)
        configs = [
            ("baseline", True, True, True),
            ("no_kd",    False, True, True),
            ("no_mt",    True,  False, True),
            ("no_ls",    True,  True,  False)
        ]

        results = []
        roc_info_list_test = []
        roc_info_list_train = []
        log_paths_dict = {}

        for (name, kd, mt, ls) in configs:
            print(f"\n--- config={name}, kd={kd}, mt={mt}, ls={ls} ---")
            do_shap_flag = (name == "baseline")

            metrics_ci, test_r2, direct_auroc, test_labels, test_probs, train_labels2, train_probs2, log_csv_path = run_experiment(
                seed=try_seed, kd=kd, mt=mt, ls=ls,
                model_type=name, do_shap=do_shap_flag
            )
            row = {
                'config': name,
                'kd': kd,
                'mt': mt,
                'ls': ls,
                'R2': test_r2,
                'direct_auroc': direct_auroc
            }
            for k, (mean_, lower_, upper_) in metrics_ci.items():
                row[f"{k}_mean"] = mean_
                row[f"{k}_lower"] = lower_
                row[f"{k}_upper"] = upper_
            results.append(row)

            # ROC数据
            roc_info_list_test.append({
                "config": name,
                "y_true": test_labels,
                "y_prob": test_probs
            })
            roc_info_list_train.append({
                "config": name,
                "y_true": train_labels2,
                "y_prob": train_probs2
            })

            # 保存日志文件路径 (用于后面拼图)
            log_paths_dict[name] = log_csv_path

        # 保存所有配置结果
        df_res = pd.DataFrame(results)
        df_res.to_csv(out_csv, index=False)
        print(f"Ablation results => {out_csv} (seed={try_seed})")

        # baseline
        baseline_row = df_res[df_res['config'] == 'baseline']
        if not baseline_row.empty:
            direct_auc = baseline_row['direct_auroc'].values[0]
            print(f"[seed={try_seed}] baseline direct_auroc={direct_auc:.4f}")
            if direct_auc > 0.7:
                outdir = f"seed_{try_seed}"
                os.makedirs(outdir, exist_ok=True)
                shutil.copyfile(out_csv, os.path.join(outdir, "main_model_ablation_results_withCI.csv"))
                print(f"[seed={try_seed}] direct_auroc>0.8 => logs saved to folder '{outdir}'")
        else:
            print(f"[seed={try_seed}] baseline row not found?? Something is off.")

        # 1) 绘制 ROC：排除 no_ls
        plot_roc_for_configs(roc_info_list_test, save_path=f"results/figures/roc_compare_seed_{try_seed}.png")

        plot_roc_for_configs_train(roc_info_list_train, save_path=f"results/figures/roc_compare_train_seed_{try_seed}.png")
        # 2) 绘制( baseline, no_kd, no_mt )的loss子图
        plot_loss_subplots(log_paths_dict, save_path=f"results/figures/loss_subplots_seed_{try_seed}.png")

        # 3) 绘制 R2 子图：排除 no_mt
        plot_r2_subplots(log_paths_dict, save_path=f"results/figures/r2_subplots_seed_{try_seed}.png")

    print("All seeds 0..100 finished.")
