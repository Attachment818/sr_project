"""Read-only descriptor-margin audit for selected FIMD pairs."""
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np, torch, yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import Predictor
from common.eval_util import list_fimd_pairs

def source(v):
    label, path = v.split('=', 1)
    return label, Path(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', action='append', required=True, type=source)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--pair', action='append', default=['39_r_t', '40_r_t'])
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for label, cfg_path in a.source:
        cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8')); cfg['PREDICT']['device']=a.device; pred=Predictor(cfg)
        pairs={x['pair_name']:x for x in list_fimd_pairs(cfg['FIMD']['data_root'])}
        for pair_id in a.pair:
            item=pairs[pair_id]; q,r=pred.image_read(item['query_im_path'], item['refer_im_path'])
            q_t=pred.trasformer(Image.fromarray(q)); r_t=pred.trasformer(Image.fromarray(r))
            k,d=pred.model_run_pair(q_t,r_t); qd=d[0].permute(1,0).numpy(); rd=d[1].permute(1,0).numpy()
            dist=np.linalg.norm(qd[:,None]-rd[None],axis=2); order=np.argsort(dist,axis=1)
            best=dist[np.arange(len(qd)),order[:,0]]; second=dist[np.arange(len(qd)),order[:,1]]
            back=np.argmin(dist,axis=0); mutual=back[order[:,0]]==np.arange(len(qd)); ratio=best/(second+1e-12)
            for name, mask in [('all',np.ones(len(qd),bool)),('ratio',ratio<cfg['PREDICT']['knn_thresh']),('mutual_ratio',(ratio<cfg['PREDICT']['knn_thresh'])&mutual)]:
                vals=ratio[mask]; rows.append({'method':label,'pair_id':item['file_name'],'stage':name,'count':int(mask.sum()),'ratio_mean':float(vals.mean()) if len(vals) else None,'ratio_p90':float(np.quantile(vals,.9)) if len(vals) else None})
    with (a.output_dir/'descriptor_margin_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (a.output_dir/'descriptor_margin_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
