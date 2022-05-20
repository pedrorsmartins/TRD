

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
import argparse
import cgi
import datetime
import os
import shutil
import urllib.request


def get_data_met(client_id, source, elements, start_date, end_date):
  
    # Define endpoint and parameters
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
        'sources': source,
        'elements': ','.join(elements),
        'referencetime': start_date.tz_localize('Europe/Oslo').tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S') + '/' + end_date.tz_localize('Europe/Oslo').tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S'),
        'timeresolutions':'PT1H',
    }
    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])

    # Convert json to dataframe format
    df = pd.DataFrame()
    print('len data = {}'.format(len(data)))
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        df = df.append(row)

    df = df.reset_index()

    # Convert the time value to something Python understands
    df['referenceTime'] = pd.to_datetime(df['referenceTime'],utc=True).dt.tz_convert('Europe/Oslo')
    df=df.set_index('referenceTime')
    df.index=df.index.tz_localize(None)

    output = pd.DataFrame()
    output['Time'] = pd.date_range(start_date,end_date,freq='H',closed='left')
    output = output.set_index('Time')
    #output['Time'] = df['referenceTime'].unique()
    #output.set_index('Time',inplace=True)
    output.index = output.index.tz_localize(None)
    for e in elements:
        #output[e] = df['value'][df['elementId']==e].values
        output = output.join(df[df['elementId']==e]['value'])
        output[e] = output['value']
        output=output.drop(['value'],axis=1)
    
    return output

def get_data_nilu(start_time,end_time,sensors,components):
    output = pd.DataFrame()
    output['Time'] = pd.date_range(start_time,end_time,freq='H',closed='right')
    output = output.set_index('Time')
    for s in sensors:
    #Sanity check because name might differ
        if s == 'torget':
            s_in = 'torvet'
        else:
            s_in = s
            
        #Need to encode sensor name because of special characters (å,ø,...)
        s_in = urllib.parse.quote(s_in)
            
        for p in components:
            # One more sanity check, as the input might come differently
            if p == 'pm25':
                p_in = 'pm2.5'
            else:
                p_in = p
                
            URL = 'https://api.nilu.no/obs/historical/{0}%20{1}/{2}%20{3}/{4}?components={5}'
    
            d1 = start_time.tz_localize('Europe/Oslo').tz_convert('Etc/GMT-1').strftime('%Y-%m-%d')
            h1 = start_time.tz_localize('Europe/Oslo').tz_convert('Etc/GMT-1').strftime('%H:%M')
            d2 = end_time.tz_localize('Europe/Oslo').tz_convert('Etc/GMT-1').strftime('%Y-%m-%d')
            h2 = end_time.tz_localize('Europe/Oslo').tz_convert('Etc/GMT-1').strftime('%H:%M')
        
            url = URL.format(d1,h1,d2,h2,s_in,p_in)

            data = pd.DataFrame.from_records(pd.read_json(url)['values'][0])
            data['toTime'] = pd.to_datetime(data['toTime'],utc=False).dt.tz_convert('Europe/Oslo')
            
            data  = data.set_index('toTime')
            data.index = data.index.tz_localize(None)
        
            output = output.join(data['value'])
            output[s+'_'+p] = output['value']
            output=output.drop(['value'],axis=1)
            
    ind = [i for i, x in enumerate(output.index.duplicated()) if x]
    output = output.drop(output.index[ind])
      
    return output


url_traffic = "https://www.vegvesen.no/trafikkdata/api/export?from={startDate}&resolution={timespan}&to={endDate}&trpIds={id}"

def get_data_traffic_csv(sensor_ids, start_date, end_date, timespan, only_total=False):
    df = {}
    
    #Read aux file with location of all sensors within the city limits
    sensor_loc = pd.read_csv('data/traffic_sensor_location.csv',delimiter=';')
    
    df_out = pd.DataFrame()
    df_out['Time'] = pd.date_range(start_date,end_date,freq='H',closed='right')
    df_out.set_index('Time',inplace=True)

    for sensor in sensor_ids:
        sensor_url = url_traffic.format(timespan=timespan,startDate=start_date.strftime('%Y-%m-%d'), endDate=end_date.strftime('%Y-%m-%d'), id=sensor)
        
        response = urllib.request.urlopen(sensor_url)

        content_disp = response.info()["Content-Disposition"]
        value, params = cgi.parse_header(content_disp)
        filename = params["filename"]
        with response, open(filename, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        
        df[sensor] = pd.read_csv(filename, sep=";", header=0, encoding="unicode_escape")
        
        #Script deletes downloaded files
        os.remove(filename) 
        
        ####Clean periods with no readings
        if df[sensor]['Volum'].dtypes != 'int64':
            df[sensor]['Volum'] = df[sensor]['Volum'].str.replace('-', '0', regex=False)
            df[sensor]['Volum'].fillna(0, inplace=True)
            df[sensor]['Volum'] = df[sensor]['Volum'].astype(int)
    
        ###########################################
        sensor_name = sensor_loc[sensor_loc['Detector ID']==sensor]['Detector Name'].unique()[0]

        df[sensor]['Til'] = pd.to_datetime(df[sensor]['Til'], utc=True).dt.tz_convert('Europe/Oslo').dt.tz_localize(None)
        df[sensor].set_index('Til',inplace=True)
    
        #in this case only output total traffic and not splitted by direction
        if only_total == False:
            directions = sensor_loc[sensor_loc['Detector ID']==sensor]['Direction'].unique()
        
            df_out = df_out.join(df[sensor][df[sensor]['Felt'].str.endswith(directions[0])]['Volum'])
            df_out.rename(columns={'Volum':sensor+'_'+directions[0]}, inplace=True)
        
            df_out = df_out.join(df[sensor][df[sensor]['Felt'].str.endswith(directions[1])]['Volum'])
            df_out.rename(columns={'Volum':sensor+'_'+directions[1]}, inplace=True)
        
        df_out = df_out.join(df[sensor][df[sensor]['Felt'] == 'Totalt']['Volum'])
        df_out.rename(columns={'Volum':sensor+'_'+'Total'}, inplace=True)
        
    return df_out    

micro_ids = {
    ##########################
    #Central Trondheim
    'sverresborg skole':'17dh0cf43jg77j',
    'ila skole':'17dh0cf43jg88e',
    'torget':'17dh0cf43jg89n',
    'elgeseter':'17dh0cf43jg89l',
    'singsaker skole':'17dh0cf43jg781',
    'berg skole':'17dh0cf43jg783',
    'bispehaugen skole':'17dh0cf43jg77n',
    'lilleby skole':'17dh0cf43jg887',
    'strindheim skole':'17dh0cf43jg889',
    ##########################
    # Eastern Trondheim
    'utleira skole':'17dh0cf43jg88b',
    'åsvang skole':'17dh0cf43jg897',
    'angeltrøa barnehage':'17dh0cf43jg89j',
    'charlottenlund barnehage':'17dh0cf43jg89f',
    'fus barnhage':'17dh0cf43jg885',
    ##########################
    #Tiller
    'åsheim skole':'17dh0cf43jg881',
    'huseby skole':'17dh0cf43jg88k',
    'sandbakken skole':'17dh0cf43jg89b',
    'tiller hvs':'17dh0cf43jg88i',
    'sjetne skole':'17dh0cf43jg899',
    ##########################
    #Tanem and Hesteskoen
    'tanem skole':'17dh0cf43jg87n',
    'hesteskoen barnehage':'17dh0cf43jg89h',
    ##########################
    #Ugla
    'ugla skole':'17dh0cf43jg77l',
    ##########################
    #Spongdal
    'spongdal':'17dh0cf43jg88g',
            }
#########################################

def get_data_micro_sensors(start_time, end_time, sensors, components):
    df_tmp = pd.read_csv('data/esp8266-240636.csv',delimiter=',')

    sensors = ['elgeseter']
    components = ['SDS011 PM2.5']

    df_iot = pd.DataFrame()
    df_iot['Time']=pd.date_range(start_time,end_time,freq='H',closed='right')
    df_iot.set_index('Time',inplace=True)

    for sensor_name in sensors:
        df_tmp['Time'] = pd.to_datetime(df_tmp['Time'])
        df_tmp.set_index('Time',inplace=True)
        #df_tmp.index = df_tmp.index.tz_localize(None)

        if start_time != None:
            df_tmp = df_tmp[df_tmp.index >= start_time]

        if end_time != None:
            df_tmp = df_tmp[df_tmp.index <= end_time]

        #Aggregate by hour (with average)
        df_tmp = df_tmp[components].resample('H',label='right').mean()
        for component in components:
            df_iot[sensor_name+'_'+component] = df_tmp[component]

        components = ['DHT22 temperature','DHT22 humidity']
        df_temp_hum = pd.read_csv('data/esp8266-244085_temp.csv',delimiter=',')

        df_temp_hum = df_temp_hum.merge(pd.read_csv('data/esp8266-244085_hum.csv',delimiter=','),on='Time', how='left')
        df_temp_hum['Time'] = pd.to_datetime(df_temp_hum['Time'])
        df_temp_hum = df_temp_hum.set_index('Time').resample('H').mean()
        df_temp_hum = df_temp_hum[components].resample('H',label='right').mean()

        df_iot = df_iot.merge(df_temp_hum,on='Time', how='left')
        
        for component in components:
            df_iot[sensor_name+'_'+component] = df_temp_hum[component]
                  
        
    return df_iot


start_time = pd.Timestamp(year=2022,month=3,day=11,hour=12)
end_time = pd.Timestamp(year=2022,month=4,day=26,hour=11)


df = pd.DataFrame()
df['Time'] = pd.date_range(start_time,end_time,freq='H',closed='right')
df = df.set_index('Time')

sensors = ['elgeseter']
components=['pm10']

df_nilu = get_data_nilu(start_time,end_time,sensors,components)

for component in components:
    for sensor in sensors:
        df[sensor+'_'+component+'_nilu'] = df_nilu[sensor+'_'+component]


# df_iot = get_data_micro_sensors(sensors=sensors,components=components, start_time=start_time, end_time=end_time)

# components = ['SDS011 PM2.5','DHT22 temperature','DHT22 humidity']

# for component in components:
#     for sensor in sensors:
#         df[sensor+'_'+component+'_iot'] = df_iot[sensor+'_'+component]

# #There are some erroneous temperature measurements from the board. Replace all unrealistic values (let's say higher than 100) with NaNs
# for sensor in sensors:
#     df.loc[df[sensor+'_DHT22 temperature_iot'] > 100,sensor+'_DHT22 temperature_iot'] = None




client_id = 'fbd8dc01-4c48-420f-9f66-79ffe28a71bb'

source = 'SN68860'
elements = ['air_temperature','relative_humidity','sum(precipitation_amount PT1H)','surface_air_pressure','wind_speed','wind_from_direction'] #['surface_air_pressure','air_temperature','relative_humidity','sum(precipitation_amount PT1H)','wind_speed','wind_from_direction']

df = df.join(get_data_met(client_id,source, elements, start_time, end_time))

df.rename(columns={"surface_air_pressure": "air_pressure", "sum(precipitation_amount PT1H)": "precipitation", "wind_from_direction": "wind_direction"}, inplace=True)


sensor_ids=['16219V72812','44656V72812','10236V72161']

#df = df.join(get_data_traffic_csv(sensor_ids, start_time, end_time,timespan='HOUR',only_total=True))


df.to_csv('data/32dump_with_all_features.csv')



print('Hello')