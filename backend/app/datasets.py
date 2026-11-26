from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
NSL_FEATURES = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
NSL_ATTACKS = {
'normal':'Normal','back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS','teardrop':'DoS','mailbomb':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS','worm':'DoS',
'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe','mscan':'Probe','saint':'Probe',
'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L','phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L','sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L','xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R','ps':'U2R','sqlattack':'U2R','xterm':'U2R'}
UNSW_ATTACKS = {'Normal':'Normal','Fuzzers':'Fuzzers','Analysis':'Analysis','Backdoors':'Backdoors','Backdoor':'Backdoors','DoS':'DoS','Exploits':'Exploits','Generic':'Generic','Reconnaissance':'Reconnaissance','Shellcode':'Shellcode','Worms':'Worms'}

def dataset_files(name):
    folder = DATA / ('nsl_kdd' if name == 'NSL-KDD' else 'unsw_nb15')
    if not folder.exists(): return []
    return sorted([p for p in folder.glob('*') if p.suffix.lower() in ('.csv','.txt') and p.is_file() and not any(token in p.name.lower() for token in ('features','events','ground_truth','_gt'))])

def load_nsl(path):
    # NSL-KDD .txt files are headerless. Always read them with header=None first;
    # pandas' default header inference would silently discard the first traffic row.
    raw = pd.read_csv(path, header=None, low_memory=False)
    if len(raw.columns) in (42, 43):
        first = str(raw.iloc[0, 0]).strip().lower() if len(raw) else ''
        if first == 'duration':
            # A headered CSV variant: re-read using its real header.
            df = pd.read_csv(path, low_memory=False)
            df.columns = [str(c).strip().lower() for c in df.columns]
            return df
        names = NSL_FEATURES + (['label', 'difficulty'] if len(raw.columns) == 43 else ['label'])
        raw.columns = names
        return raw
    # Fallback for a headered CSV with a non-standard column count.
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def load_unsw(path):
    df=pd.read_csv(path, low_memory=False)
    df.columns=[str(c).strip() for c in df.columns]
    return df

def load_dataset(name, path=None):
    if path:
        p=Path(path)
        return load_nsl(p) if name == 'NSL-KDD' else load_unsw(p)
    files=dataset_files(name)
    if not files: raise FileNotFoundError(f'No {name} files found. Run python scripts/setup_datasets.py or add official CSV files to backend/data.')
    frames=[]
    for f in files:
        if name == 'NSL-KDD': frames.append(load_nsl(f))
        else: frames.append(load_unsw(f))
    return pd.concat(frames, ignore_index=True)

def target_column(df, name):
    candidates = (['attack_cat','attack_category','label',' Label','class'] if name == 'UNSW-NB15' else ['label',' Label','attack_cat','attack_category','class'])
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f'Could not identify target column. Expected one of: {candidates}')

def prepare_labels(df, name):
    col=target_column(df,name)
    raw=df[col].astype(str).str.strip().str.rstrip('.')
    if name == 'NSL-KDD':
        y=raw.str.lower().map(NSL_ATTACKS).fillna(raw.str.lower().map(lambda x: 'Normal' if x in ('normal','0') else 'Unknown'))
        y=y.replace({'Unknown':'Other'})
    else:
        if col == 'label':
            y=raw.map(lambda x: 'Normal' if x.lower() in ('normal','0') else 'Attack')
        else:
            y=raw.map(lambda x: 'Normal' if x.lower() in ('normal','0') else (UNSW_ATTACKS.get(x, x)))
    return col, raw, y

def feature_frame(df, name, label_col):
    excluded={label_col,'label','difficulty','attack_cat','attack_category','class','id','srcip','dstip','stime','ltime'}
    if name == 'NSL-KDD': excluded |= {'difficulty'}
    cols=[c for c in df.columns if c not in excluded]
    if not cols: raise ValueError('No usable feature columns found.')
    return df[cols].copy()

def status(name):
    files=dataset_files(name)
    return {'dataset':name,'available':bool(files),'files':[p.name for p in files], 'model_ready':(ROOT/'artifacts'/name.lower().replace('-','_')/'metadata.json').exists()}
