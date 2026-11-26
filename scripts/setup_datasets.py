"""Fetches NSL-KDD from a public mirror and prints official UNSW-NB15 preparation steps.
The UNSW official distribution is hosted behind a SharePoint download page; no unverified
mirror is silently substituted. Users can pass direct CSV URLs via environment variables.
"""
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import hashlib, os, sys, io, zipfile
ROOT=Path(__file__).resolve().parents[1]
NSL=ROOT/'backend'/'data'/'nsl_kdd'; UNSW=ROOT/'backend'/'data'/'unsw_nb15'
NSL.mkdir(parents=True,exist_ok=True); UNSW.mkdir(parents=True,exist_ok=True)

EXPECTED_SHA256={
    'KDDTrain+.txt':'1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95',
    'KDDTest+.txt':'fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84',
    'UNSW_NB15_training-set.csv':'bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa',
    'UNSW_NB15_testing-set.csv':'734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559',
    'UNSW-NB15_features.csv':'c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce',
}
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def valid_file(path):
    expected=EXPECTED_SHA256.get(path.name)
    return path.is_file() and path.stat().st_size>100_000 and (not expected or sha256(path)==expected)

def download(url,dest):
    print(f'GET {url}')
    req=Request(url,headers={'User-Agent':'SENTINEL academic dataset setup/1.0'})
    with urlopen(req,timeout=90) as r, open(dest,'wb') as f:
        while True:
            block=r.read(1024*1024)
            if not block: break
            f.write(block)
    if dest.stat().st_size < 100_000:
        dest.unlink(missing_ok=True); raise RuntimeError('Downloaded file is unexpectedly small; rejected.')
    h=hashlib.sha256(dest.read_bytes()).hexdigest(); print(f'OK {dest.name} ({dest.stat().st_size:,} bytes) sha256={h}')

print('\nSENTINEL dataset setup — genuine sources only\n')
# Bundled datasets take precedence. Validate the expected train/test pair before
# considering any network download; no setup-time download is needed for this package.
unsw_pair=('UNSW_NB15_training-set.csv','UNSW_NB15_testing-set.csv')
if all(valid_file(UNSW/n) for n in unsw_pair):
    print('SKIP UNSW-NB15: bundled training/testing CSVs passed SHA-256 verification.')
    for n in unsw_pair: print(f'VERIFIED {n} sha256={sha256(UNSW/n)}')
else:
    print('UNSW-NB15 pair incomplete; attempting official SharePoint setup if possible.')
nsl_urls={
'KDDTrain+.txt':'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
'KDDTest+.txt':'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'}
for name,url in nsl_urls.items():
    dest=NSL/name
    if valid_file(dest):
        print(f'SKIP {dest.name}: SHA-256 verified ({sha256(dest)})'); continue
    try: download(url,dest)
    except Exception as e: print(f'FAILED {name}: {e}')
# Try the official UNSW SharePoint folder's download endpoint. Some networks or
# SharePoint policies redirect to Microsoft login; in that case we report it and
# leave the user with the official browser download instructions below.
sharepoint='https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS&download=1'
if not all(valid_file(UNSW/n) for n in unsw_pair):
    try:
        req=Request(sharepoint,headers={'User-Agent':'Mozilla/5.0 SENTINEL academic dataset setup/1.0'})
        with urlopen(req,timeout=90) as response: payload=response.read()
        if not payload.startswith(b'PK\x03\x04'):
            raise RuntimeError('Official SharePoint did not return a ZIP archive (possibly requires browser sign-in).')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            wanted={'UNSW_NB15_training-set.csv','UNSW_NB15_testing-set.csv'}
            found=set(archive.namelist())
            for member in archive.namelist():
                base=Path(member).name
                if base in wanted:
                    target=UNSW/base; target.write_bytes(archive.read(member))
                    print(f'OK {base} ({target.stat().st_size:,} bytes) sha256={hashlib.sha256(target.read_bytes()).hexdigest()}')
        if not all(valid_file(UNSW/n) for n in unsw_pair):
            raise RuntimeError('Official archive did not contain both expected training/testing CSV files.')
    except Exception as e: print(f'UNSW official automatic download unavailable: {e}')

# Optional direct UNSW URLs may be supplied after obtaining them from UNSW's official download.
for key,name in [('UNSW_TRAIN_URL','UNSW_NB15_training-set.csv'),('UNSW_TEST_URL','UNSW_NB15_testing-set.csv')]:
    url=os.environ.get(key)
    if url:
        try: download(url,UNSW/name)
        except Exception as e: print(f'FAILED {name}: {e}')
    else: print(f'UNSW: {key} not set; no direct URL supplied (bundled files remain in use if present).')
print('\nUNSW-NB15 official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset')
if not all(valid_file(UNSW/n) for n in unsw_pair):
    print('If unavailable, use the official SharePoint page to obtain UNSW_NB15_training-set.csv and')
    print('UNSW_NB15_testing-set.csv, then place both in backend/data/unsw_nb15/.')
print('\nDataset availability:')
print('NSL-KDD:', 'READY' if all(valid_file(NSL/n) for n in ('KDDTrain+.txt','KDDTest+.txt')) else 'MISSING OR CHECKSUM MISMATCH')
print('UNSW-NB15:', 'READY' if all((UNSW/n).is_file() and (UNSW/n).stat().st_size > 1_000_000 for n in unsw_pair) else 'MISSING')
