"""三教师离线请求预览：构造真实PNG字节的请求，但不读密钥、不联网。"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BLIND=ROOT/'协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/blind_inputs.jsonl'
PROMPT=ROOT/'协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt'
CHECKER=ROOT/'协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py'
EXPECTED={BLIND:'70d48e6f5a7ab3c14f4c3d675e94adc11a9cb518f814120fadb8b2e802120b20',PROMPT:'7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a',CHECKER:'794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55'}
CANDIDATES={
 'kimi':{'provider':'bailian','model':'kimi/kimi-k3','endpoint':'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions','credential_env':'DASHSCOPE_API_KEY'},
 'qwen':{'provider':'bailian','model':'qwen3.8-max','endpoint':'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions','credential_env':'DASHSCOPE_API_KEY'},
 'deepseek':{'provider':'deepseek_official','model':'deepseek-flash','endpoint':'https://api.deepseek.com/chat/completions','credential_env':'DEEPSEEK_API_KEY'}}

def preview(limit):
    for p,h in EXPECTED.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h, '固定输入漂移'
    blind=[json.loads(l) for l in BLIND.read_text(encoding='utf-8').splitlines() if l]
    assert len(blind)==18 and len({r['sample_id'] for r in blind})==18
    prompt=PROMPT.read_bytes().decode('utf-8')
    rows=[]
    for b in blind[:limit]:
        assert set(b)=={'item_id','sample_id','image_path','image_sha256'}
        raw=Path(b['image_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==b['image_sha256']
        assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        url='data:image/png;base64,'+base64.b64encode(raw).decode('ascii')
        assert base64.b64decode(url.split(',',1)[1],validate=True)==raw
        for name,config in CANDIDATES.items():
            body={'model':config['model'],'messages':[{'role':'user','content':[
                {'type':'image_url','image_url':{'url':url}}, {'type':'text','text':prompt}]}],
                'max_tokens':16384,'stream':True}
            serialized=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode('utf-8')
            rows.append({'candidate':name,**config,'item_id':b['item_id'],'sample_id':b['sample_id'],
                         'image_sha256':b['image_sha256'],'prompt_byte_sha256':EXPECTED[PROMPT],
                         'payload_sha256':hashlib.sha256(serialized).hexdigest(),'image_bytes':len(raw),
                         'payload_bytes':len(serialized),'messages':1,'chat_history_included':False,
                         'max_tokens':16384,'stream':True,'client_retries':0,'timeout_seconds':600,
                         'image_bytes_roundtrip_verified':True,'budget_and_price_verified':False,
                         'model_vision_endpoint_tested':False,'request_sent':False})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=18)
    ap.add_argument('--write-preview',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if not 1<=a.limit<=18:ap.error('limit必须为1..18')
    rows=preview(a.limit)
    if a.write_preview:
        path=HERE/('three_teachers_preview_'+str(a.limit)+'.jsonl')
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode('utf-8')
        if path.exists():
            if not(a.resume and path.read_bytes()==data):raise ValueError('拒绝覆盖预览文件')
        else:path.write_bytes(data)
    print(json.dumps({'preview_requests':len(rows),'unique_images':a.limit,'candidates':list(CANDIDATES),'secret_reads':0,'network_requests':0,'image_bytes_verified':True,'paid_test_status':'not run'},ensure_ascii=False))

if __name__=='__main__':main()
