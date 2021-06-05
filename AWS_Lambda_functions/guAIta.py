import json
import sys
import logging
import struct
import codecs
import os
from datetime import datetime, date
from dateutil import tz
import time
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import binascii
from math import pi
import base64
import requests

s3 = boto3.resource("s3")

def lambda_handler(event, context):
    if "log" in event:
        if event['log']=='Start':
            telegram_message = event['obs_id'] + " start scanning Images! "
            requests.get("https://api.telegram.org/botxxxxxxxxxxxxx:XXXXXXXXXXXXXXXX/sendMessage?text="+telegram_message+"&chat_id=XXXXXXXXX")    
        else:
            telegram_message = event['obs_id'] + " finish scan! Total images : "+event['total']
            requests.get("https://api.telegram.org/botxxxxxxxxxxxxx:XXXXXXXXXXXXXXXX/sendMessage?text="+telegram_message+"&chat_id=XXXXXXXXX")    
    else:
        id_envio=event['obs_id']+"_"+event['id']
    
        folder_json = "detections"
        folder_img  = "imgs"
        folder_day  = event['folder']
        file_name_json = folder_json + "/" + folder_day + "/" + id_envio + ".json"
        file_name_img  = folder_img + "/" + folder_day + "/" + id_envio + ".jpeg"
        img = base64.b64decode(event['img'])
        s3.Bucket("guaita").put_object(Key=file_name_img,ContentType="image/jpeg", Body=img)
        
        event.pop('img',None)
        s3.Bucket("guaita").put_object(Key=file_name_json, Body=json.dumps(event).encode())
        
        if (event['telegram']=="True"):
            telegram_message = event['obs_id'] + " : "+ event['date'] + " - " + event['time'] + "\n" + "https://guaita.s3-eu-west-1.amazonaws.com/imgs/"+folder_day+"/"+id_envio+".jpeg"+"?a="+str(time.time())
            requests.get("https://api.telegram.org/botxxxxxxxxxxxxx:XXXXXXXXXXXXXXXX/sendMessage?text="+telegram_message+"&chat_id=XXXXXXXXX")
    
    
    return {'statusCode': 200, 'body': json.dumps({'message': 'successful lambda function call'}), 'headers': {'Access-Control-Allow-Origin': '*'}}     