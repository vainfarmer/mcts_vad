from pathlib import Path
import pandas as pd, json, cv2
root=Path('F:/dataset/ucf_crime');out=Path(__file__).resolve().parent
ann={}
for line in (root/'UCF_Crimes-Train-Test-Split/Temporal_Anomaly_Annotation_for_Testing_Videos.txt').read_text().splitlines():
 p=line.split()
 if p:ann[p[0]]=(p[1],list(map(int,p[2:6])))
files={}
for p in root.rglob('*.mp4'):
 if 'test' not in p.relative_to(root).parts:files.setdefault(p.name,[]).append(p)
rows=[]
for line in (root/'UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt').read_text().splitlines():
 name=Path(line.strip()).name
 if not name:continue
 matches=files.get(name,[])
 if not matches:raise ValueError('Missing video '+name)
 cat='Normal' if 'normal' in line.lower() else line.split('/')[0]
 vals=ann[name][1] if cat!='Normal' else [-1]*4
 rows.append(dict(video_path=str(sorted(matches)[0].resolve()),category=cat,**dict(zip(['annotation_start1','annotation_end1','annotation_start2','annotation_end2'],vals))))
frame=pd.DataFrame(rows);target=out/'dataset/data';target.mkdir(parents=True,exist_ok=True);frame.to_parquet(target/'test-00000-of-00001.parquet',index=False)
selected=pd.concat([g.sample(n=1,random_state=42).sort_values('video_path') for _,g in frame.groupby('category')]);records=[]
for r in selected.to_dict('records'):
 cap=cv2.VideoCapture(r['video_path']);r['fps']=cap.get(cv2.CAP_PROP_FPS);r['frames']=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();assert r['frames']>0;r['duration']=r['frames']/r['fps'];records.append(r)
(out/'selected_videos.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('test_rows',len(frame),'categories',len(records));print([(r['category'],Path(r['video_path']).name,round(r['duration']),r['fps']) for r in records])
