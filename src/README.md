# Setup
```shell
pip install -r requirements.txt
```

# Training
```shell
python train.py
```
- Example output: 
```
best checkpoint /Users/tunv/Workspace/Security Projects/WAD-Bert/checkpoints/bert-tiny-web-attack-classification-epoch=00-val_loss=0.00.ckpt
```

# Inference
- Inference by model name, request and device
```shell
python inference.py --model "./../checkpoints/bert-tiny-web-attack-classification-epoch=00-val_loss=0.00.ckpt" --request "GET /index.html HTTP/1.1" --device "cpu"
```
- Example output:
```shell
{'class_idx': 1, 'class_name': 'benign'} duration 10.766785383224487 sec(s) per 1000 predictions
```

# API
Deploy as service at start (use uvicorn)
```bash
cat /etc/systemd/system/uvicorn.service
sudo chmod 644 /etc/systemd/system/uvicorn.service
sudo systemctl daemon-reload
sudo systemctl start uvicorn
sudo systemctl enable uvicorn
sudo systemctl status uvicorn

sudo systemctl stop uvicorn
```

### Go to [http://103.192.237.84:8000/docs](http://103.192.237.84:8000/docs) and try with a `request content` at /predict route

Example 1:
```http
INPUT: 

GET /openautoclassifieds/friendmail.php?listing=<script>alert(document.domain);</script> HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Connection: close

-----------------
OUTPUT:

{
  "class_idx": 0,
  "class_name": "anomaly"
}
```

Example 2:
```http
INPUT: 

GET http://localhost:8080/tienda1/publico/autenticar.jsp?modo=entrar&login=gelais&pwd=coj*itranco&remember=off&B1=Entrar HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Connection: close

-----------------
OUTPUT:

{
  "class_idx": 1,
  "class_name": "benign"
}
```




