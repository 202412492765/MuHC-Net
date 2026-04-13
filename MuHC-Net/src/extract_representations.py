import pandas as pd
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ae import Encoder, AttnFusion
from config.paths import DATA_DIR, OUTPUT_DIR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
x_mir = torch.tensor(pd.read_csv(DATA_DIR / 'miRNA_preprocess.csv', index_col=0).values, 
                     dtype=torch.float32).to(device)
x_meth = torch.tensor(pd.read_csv(DATA_DIR / 'meth_preprocess.csv', index_col=0).values, 
                      dtype=torch.float32).to(device)
x_mrna = torch.tensor(pd.read_csv(DATA_DIR / 'mRNA_preprocess.csv', index_col=0).values, 
                      dtype=torch.float32).to(device)

# Load model
checkpoint = torch.load(OUTPUT_DIR / 'ae_model.pth', map_location=device)
enc_mir = Encoder(x_mir.shape[1], hid=64).to(device)
enc_meth = Encoder(x_meth.shape[1], hid=64).to(device)
enc_mrna = Encoder(x_mrna.shape[1], hid=64).to(device)
attn_fusion = AttnFusion(hid=64).to(device)

enc_mir.load_state_dict(checkpoint['enc_mir'])
enc_meth.load_state_dict(checkpoint['enc_meth'])
enc_mrna.load_state_dict(checkpoint['enc_mrna'])
attn_fusion.load_state_dict(checkpoint['attn_fusion'])

# Extract features
enc_mir.eval(); enc_meth.eval(); enc_mrna.eval(); attn_fusion.eval()
with torch.no_grad():
    s_mir, p_mir = enc_mir(x_mir)
    s_meth, p_meth = enc_meth(x_meth)
    s_mrna, p_mrna = enc_mrna(x_mrna)
    h_shared, _ = attn_fusion(s_mir, s_meth, s_mrna)

    pat_id = pd.read_csv(DATA_DIR / 'miRNA_preprocess.csv', index_col=0).index

    h_shared_df = pd.DataFrame(h_shared.cpu().numpy(), index=pat_id, columns=[f'shared_{i}' for i in range(64)])
    p_mir_df = pd.DataFrame(p_mir.cpu().numpy(), index=pat_id, columns=[f'priv_mir_{i}' for i in range(64)])
    p_meth_df = pd.DataFrame(p_meth.cpu().numpy(), index=pat_id, columns=[f'priv_meth_{i}' for i in range(64)])
    p_mrna_df = pd.DataFrame(p_mrna.cpu().numpy(), index=pat_id, columns=[f'priv_mrna_{i}' for i in range(64)])

    h_shared_df.to_csv(DATA_DIR / 'h_shared_64d.csv', index=True)
    p_mir_df.to_csv(DATA_DIR / 'p_mir_64d.csv', index=True)
    p_meth_df.to_csv(DATA_DIR / 'p_meth_64d.csv', index=True)
    p_mrna_df.to_csv(DATA_DIR / 'p_mrna_64d.csv', index=True)

print('4 feature files saved (all 64-dim)')