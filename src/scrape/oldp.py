import oldp_client 

conf = oldp_client.Configuration()

conf.api_key['Authorization'] = '73efd304f9ecb4d715768ca19523e87f86c8ec83'  # TODO Replace this with your API key
conf.api_key_prefix['Authorization'] = 'Token'

api_client = oldp_client.ApiClient(conf)

laws_api = oldp_client.LawsApi(api_client)

#courts = courts_api.courts_list().results
thread = laws_api.laws_list(async_req=True)
result = thread.get()

import pandas as pd

df = pd.DataFrame(result)
df.head()