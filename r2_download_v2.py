import requests, os, time

AUTH_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
USERNAME = 'adegaking1@gmail.com'
PASSWORD = 'Kingjones400$'
OUT_DIR = 'outputs/sentinel2/R2'
os.makedirs(OUT_DIR, exist_ok=True)

SCENES = [
    'S2B_MSIL1C_20220108T211909_N0510_R100_T04QFJ_20240425T120144.SAFE',
    'S2B_MSIL1C_20220115T210919_N0510_R057_T04QFJ_20240430T032942.SAFE',
    'S2A_MSIL1C_20220120T210921_N0510_R057_T04QFJ_20240503T032713.SAFE',
    'S2A_MSIL1C_20220123T211921_N0510_R100_T04QFJ_20240507T191612.SAFE',
    'S2B_MSIL1C_20220125T210919_N0510_R057_T04QFJ_20240511T114500.SAFE',
    'S2B_MSIL1C_20220128T211909_N0510_R100_T04QFJ_20240511T191208.SAFE',
    'S2B_MSIL1C_20220105T210919_N0510_R057_T04QGJ_20240423T132426.SAFE',
    'S2B_MSIL1C_20220115T210919_N0510_R057_T04QGJ_20240430T032942.SAFE',
    'S2A_MSIL1C_20220120T210921_N0510_R057_T04QGJ_20240503T032713.SAFE',
    'S2B_MSIL1C_20220125T210919_N0510_R057_T04QGJ_20240511T114500.SAFE',
    'S2A_MSIL1C_20220130T210921_N0510_R057_T04QGJ_20240514T073400.SAFE',
    'S2B_MSIL1C_20220204T210919_N0510_R057_T04QGJ_20240519T041947.SAFE',
    'S2A_MSIL1C_20220209T210921_N0510_R057_T04QGJ_20240519T204456.SAFE',
    'S2B_MSIL1C_20220214T210919_N0510_R057_T04QGJ_20240516T133552.SAFE',
    'S2B_MSIL1C_20220102T205939_N0510_R014_T05QKB_20240422T031149.SAFE',
    'S2B_MSIL1C_20220105T210919_N0510_R057_T05QKB_20240423T132426.SAFE',
    'S2A_MSIL1C_20220107T205941_N0510_R014_T05QKB_20240424T234517.SAFE',
    'S2A_MSIL1C_20220110T210931_N0510_R057_T05QKB_20240426T192945.SAFE',
    'S2B_MSIL1C_20220102T205939_N0510_R014_T05QKC_20240422T031149.SAFE',
    'S2B_MSIL1C_20220105T210919_N0510_R057_T05QKC_20240423T132426.SAFE',
    'S2A_MSIL1C_20220107T205941_N0510_R014_T05QKC_20240424T234517.SAFE',
    'S2A_MSIL1C_20220110T210931_N0510_R057_T05QKC_20240426T192945.SAFE',
    'S2B_MSIL1C_20220112T205939_N0510_R014_T05QKC_20240427T235153.SAFE',
    'S2B_MSIL1C_20220115T210919_N0510_R057_T05QKC_20240430T032942.SAFE',
    'S2A_MSIL1C_20220117T205941_N0510_R014_T05QKC_20240501T142547.SAFE',
    'S2B_MSIL1C_20220122T205939_N0510_R014_T05QKC_20240504T232828.SAFE',
    'S2B_MSIL1C_20220112T205939_N0510_R014_T05QLB_20240427T235153.SAFE',
    'S2A_MSIL1C_20220117T205941_N0510_R014_T05QLB_20240501T142547.SAFE',
    'S2B_MSIL1C_20220122T205939_N0510_R014_T05QLB_20240504T232828.SAFE',
    'S2A_MSIL1C_20220127T205941_N0510_R014_T05QLB_20240511T150356.SAFE',
    'S2A_MSIL1C_20220206T205941_N0510_R014_T05QLB_20240521T083027.SAFE',
    'S2B_MSIL1C_20220211T205929_N0510_R014_T05QLB_20240518T180519.SAFE',
    'S2A_MSIL1C_20220216T205941_N0510_R014_T05QLB_20240517T072240.SAFE',
    'S2B_MSIL1C_20220221T205939_N0510_R014_T05QLB_20240517T211831.SAFE',
    'S2A_MSIL1C_20220117T205941_N0510_R014_T05QLC_20240501T142547.SAFE',
    'S2B_MSIL1C_20220201T205929_N0510_R014_T05QLC_20240520T155917.SAFE',
    'S2A_MSIL1C_20220206T205941_N0510_R014_T05QLC_20240521T083027.SAFE',
    'S2B_MSIL1C_20220211T205929_N0510_R014_T05QLC_20240518T180519.SAFE',
]

def get_token():
    r = requests.post(AUTH_URL, data={
        'grant_type': 'password',
        'client_id': 'cdse-public',
        'username': USERNAME,
        'password': PASSWORD
    })
    r.raise_for_status()
    return r.json()['access_token']

def search_id(name, token):
    base = name.replace('.SAFE', '')
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{base}.SAFE'&$top=1"
    r = requests.get(url, headers={'Authorization': f'Bearer {token}'}, timeout=30)
    items = r.json().get('value', [])
    return items[0]['Id'] if items else None

success, failed = 0, 0
total = len(SCENES)

for i, scene in enumerate(SCENES):
    out_path = os.path.join(OUT_DIR, scene + '.zip')
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        if sz > 50 * 1024 * 1024:
            print(f'[{i+1}/{total}] SKIP (exists): {scene}')
            success += 1
            continue
        else:
            os.remove(out_path)
    print(f'[{i+1}/{total}] Fetching token + downloading: {scene}')
    try:
        token = get_token()
        pid = search_id(scene, token)
        if not pid:
            print(f'  NOT FOUND in catalogue: {scene}')
            failed += 1
            continue
        dl_url = f'https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value'
        r = requests.get(dl_url, headers={'Authorization': f'Bearer {token}'}, stream=True, timeout=600)
        if r.status_code == 401:
            print(f'  FAILED 401 even with fresh token: {scene}')
            failed += 1
            continue
        r.raise_for_status()
        downloaded = 0
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        size_mb = downloaded / 1024 / 1024
        print(f'  Downloaded: {scene} ({size_mb:.1f} MB)')
        success += 1
        time.sleep(3)
    except Exception as e:
        print(f'  ERROR: {e}')
        if os.path.exists(out_path):
            os.remove(out_path)
        failed += 1
        time.sleep(5)

print(f'Done. Success: {success}, Failed: {failed}')