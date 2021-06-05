import json
import boto3
from datetime import date, timedelta
import requests
from datetime import datetime


s3 = boto3.resource("s3")

def lambda_handler(event, context):
    yesterday = date.today() - timedelta(days=1)
    folder = yesterday.strftime('%Y%m%d')
    
    code = "<html><head><title>guAIta</title><link rel='stylesheet' href='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'>"
    code = code + "<link rel='canonical' href='https://getbootstrap.com/docs/4.0/examples/starter-template/'><link href='starter-template.css' rel='stylesheet'></head>"
    code = code + "<body class='starter-template'><div><img src='logo.jpeg' width='10%' heigth='10%'><p class='lead'>Automate Meteor Detection</p>"
    code = code + "<p class='lead'>Version 1.0.0 (Beta) - Created by David Regordosa @pisukeman"+ "</p></div>"
    code = code + "<p class='lead'>Day: "+yesterday.strftime('%d/%m/%Y')+"</p>"
    
    code = code +  "<a class='text-secondary' href='"+(date.today() - timedelta(days=2)).strftime('%Y%m%d')+".html'><em>"+(date.today() - timedelta(days=2)).strftime('%d/%m/%Y')+"   </em></a>"
    code = code +  "<a class='text-secondary' href='"+(date.today() - timedelta(days=3)).strftime('%Y%m%d')+".html'><em>"+(date.today() - timedelta(days=3)).strftime('%d/%m/%Y')+"   </em></a>"
    code = code +  "<a class='text-secondary' href='"+(date.today() - timedelta(days=4)).strftime('%Y%m%d')+".html'><em>"+(date.today() - timedelta(days=4)).strftime('%d/%m/%Y')+"   </em></a>"
    code = code +  "<a class='text-secondary' href='"+(date.today() - timedelta(days=5)).strftime('%Y%m%d')+".html'><em>"+(date.today() - timedelta(days=5)).strftime('%d/%m/%Y')+"   </em></a>"
    code = code +  "<a class='text-secondary' href='"+(date.today() - timedelta(days=6)).strftime('%Y%m%d')+".html'><em>"+(date.today() - timedelta(days=6)).strftime('%d/%m/%Y')+"</em></a>"

    bucket = s3.Bucket("guaita")
    
    objects = bucket.objects.filter(Prefix="detections/"+folder+"/")
    same=False
    datetime_past=datetime(1990,1,1,1,1,1)
    c=0
    for l in objects:
        if l.key=="detections/"+folder+"/":
            continue
        body = bucket.Object(l.key).get()['Body'].read()
        parsed_message = json.loads(body)
        
        c=c+1
        id = parsed_message['id']
        datetime_act =datetime(int(id[0:4]),int(id[4:6]),int(id[6:8]),int(id[8:10]),int(id[10:12]),int(id[12:14]))
        if ((datetime_act-datetime_past).total_seconds()<=35):
            same=True
        else:
            same=False
            code=code+"</br>"
            
        img_name = l.key[11:-4]+"jpeg"
        if (same==False):
            code=code+"</br>"
            code = code + "<img src='"+"/imgs/"+img_name+"'>"
            code=code+"</br>"
            code = code + "<a class='text-info' href=/imgs/"+img_name+"><em>"+parsed_message['obs_id'] + " - "+ parsed_message['date']+" - "+ parsed_message['time']+"    "+"</em></a><span class='badge badge-pill badge-primary'>  "+parsed_message['score'][0:4]+"</span>"
        else:
            code = code + "<a class='text-secondary' href=/imgs/"+img_name+"><em>"+"    "+ parsed_message['time']+"    "+"</em></a><span class='badge badge-pill badge-primary'>  "+parsed_message['score'][0:4]+"</span>"            
        
        datetime_past = datetime_act
    
    if (c==0):
        code = code + "<div class='alert alert-success' role='alert'> No detections Today </div>"
    
    code = code + "</br></div>"
    code=code+"</br></br>"
    Pujalt_iss_file = bucket.Object("Pujalt_iss.json").get()['Body'].read()
    Pujalt_iss = json.loads(Pujalt_iss_file)
    times = []
    if Pujalt_iss:
        code = code + "<div class='alert alert-info role='alert'> ISS Transits for Obs: Pujalt</br>"
        for iss in Pujalt_iss['passes']:
            times.append([datetime.fromtimestamp(iss['startUTC']).strftime("%b %d %Y %H:%M:%S"),datetime.fromtimestamp(iss['endUTC']).strftime("%b %d %Y %H:%M:%S")])
            code = code + "("+datetime.fromtimestamp(iss['startUTC']).strftime("%b %d %Y %H:%M:%S")+"-"+datetime.fromtimestamp(iss['endUTC']).strftime("%b %d %Y %H:%M:%S")+") "
        code = code + "</div>"

    code = code + "</body></html>"
    
    file_name = "index.html"
    bucket.put_object(Key=file_name, Body=code,ContentType = "text/html")
    file_name = folder+".html"
    bucket.put_object(Key=file_name, Body=code,ContentType = "text/html")

    #Pujalt
    json_iss = requests.get("https://api.n2yo.com/rest/v1/satellite//visualpasses/25544/41.718333/1.421667/770/1/5&apiKey=XXXXXXXXXXXXXXX").json()
    s3.Bucket("guaita").put_object(Key="Pujalt_iss.json", Body=json.dumps(json_iss).encode())

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
