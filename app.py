# Import required libraries
import pandas as pd
import numpy as np
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pickle

# Read data
df=pd.read_csv("Data/healthcare-dataset-stroke-data.csv")


# Create a dash application
app = dash.Dash(__name__)
server = app.server
# Build dash app layout
app.layout = html.Div(children=[ html.H1('Stroke statitics', 
                                style={'textAlign': 'center', 'color': '#503D36',
                                'font-size': 30,'backgroundColor': '#FDE3C8'}),
                                html.H2('Predict stroke', 
                                style={'textAlign': 'left', 'color': '#503D36',
                                'font-size': 24}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='gender_dd',
                                        options=[
                                            {'label': 'male','value':'Male'},
                                            {'label':'female','value':'Female'}
                                        ],
                                        value='Male',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='age_dd',
                                        options=[
                                            {'label': 'age: 12-19','value':'12-20'},
                                            {'label': 'age: 20-24','value':'20-25'},
                                            {'label': 'age: 25-59','value':'25-60'},
                                            {'label': 'age: 60-85','value':'> 60'}
                                        ],
                                        value='25-60',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='hypertension_dd',
                                        options=[
                                            {'label': 'hyper','value':'1'},
                                            {'label':'no hyper','value':'0'}
                                        ],
                                        value='1',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='heart_dd',
                                        options=[
                                            {'label': 'heart_ds','value':'1'},
                                            {'label':'no heart_ds','value':'0'}
                                        ],
                                        value='1',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='married_dd',
                                        options=[
                                            {'label': 'married','value':'Yes'},
                                            {'label':'not married','value':'No'}
                                        ],
                                        value='Yes',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='work_dd',
                                        options=[
                                            {'label': 'Private','value':'Private'},
                                            {'label': 'Self-employed','value':'Self-employed'},
                                            {'label': 'Govt_job','value':'Govt_job'},
                                            {'label': 'Children','value':'Children'},
                                            {'label':'Never_worked','value':'Never_worked'}
                                        ],
                                        value='Private',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='residence_dd',
                                        options=[
                                            {'label': 'Urban','value':'Urban'},
                                            {'label':'Rural','value':'Rural'}
                                        ],
                                        value='Urban',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='smoking_dd',
                                        options=[
                                            {'label': 'formerly smoked','value':'formerly smoked'},
                                            {'label': 'smokes','value':'smokes'},
                                            {'label': 'Unknown','value':'Unknown'},
                                            {'label':'never smoked','value':'never smoked'}
                                        ],
                                        value='smokes',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='glu_dd',
                                        options=[
                                            {'label': 'gluc: 0-139','value':'0-139'},
                                            {'label': 'gluc: 140-199','value':'140-199'},
                                            {'label':'gluc: >200','value':'> 200'}
                                        ],
                                        value='140-199',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    ),
                                    dcc.Dropdown(
                                        id='bmi_dd',
                                        options=[
                                            {'label': 'bmi: 0-18.5','value':'1'},
                                            {'label': 'bmi: 18.5-25','value':'2'},
                                            {'label': 'bmi: 25-30','value':'3'},
                                            {'label': 'bmi: 30-35','value':'4'},
                                            {'label': 'bmi: 35-40','value':'5'},
                                            {'label':'bmi: >40','value':'6'}
                                        ],
                                        value='2',
                                        searchable=False,
                                        style={'height': '30px', 'width': '140px'}
                                    )
                                ],style={'display': 'flex'}),
                                html.Br(),
                                html.Div(id='display_predict'
                                    ,style={'textAlign': 'center', 'color': '#484E5D',
                                    'font-size': 22,'backgroundColor': '#FFB6B6'}),
                                html.Br(),
                                html.Br(), 
                                # Segment 1
                                html.Div([
                                        html.Div(dcc.Graph(id='stroke-plot'), style={"border":"1px black solid"}),
                                        html.Div(dcc.Graph(id='ss-plot'), style={"border":"1px black solid"})
                                ], style={'display': 'flex'}),
                                ])



def compute_bmi(df):
    df_q1=df[df['gender']!='Other']
    df_q1=df_q1[df_q1['bmi'].notnull()]
    bins=[0,18.5,25,30,35,40,100]
    labels=[1,2,3,4,5,6]
    df_q1['bin']=pd.cut(df_q1['bmi'],right=False,bins=bins,labels=labels)
    df_gen_bin=df_q1.groupby(["bin","gender"])['stroke'].mean()*100
    df_gen_bin=df_gen_bin.unstack(1)
    return df_gen_bin

def preprocess_2(df):
    df_temp=df.groupby(['work_type','smoking_status'])['stroke'].value_counts()
    df_temp=df_temp.to_frame()
    df_temp.columns=['number']
    df_temp.reset_index(inplace=True)
    return df_temp

# Callback decorator
@app.callback([
                Output(component_id='display_predict', component_property='children'),
                Output(component_id='stroke-plot', component_property='figure'),
                Output(component_id='ss-plot', component_property='figure'),
                ],
                [
                Input(component_id='gender_dd', component_property='value'),
                Input(component_id='age_dd', component_property='value'),
                Input(component_id='hypertension_dd', component_property='value'),
                Input(component_id='heart_dd', component_property='value'),
                Input(component_id='married_dd', component_property='value'),
                Input(component_id='work_dd', component_property='value'),
                Input(component_id='residence_dd', component_property='value'),
                Input(component_id='smoking_dd', component_property='value'),
                Input(component_id='glu_dd', component_property='value'),
                Input(component_id='bmi_dd', component_property='value'),
                ])
# Computation to callback function and return graph
def get_graph(gender,age,hyper,heart,married,work,residence,smoke,glu,bmi):
    data_list=[[gender,hyper,heart,married,work,residence,smoke,bmi,age,glu]]
    df_predict=pd.DataFrame(data_list,columns=['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status','bmi_bin','age_bin','glu_bin'])
    with open('Data/mlp.pkl','rb') as f:
        loaded_model = pickle.load(f)
        result = loaded_model.predict_proba(df_predict)*100
    result2=np.round(result[0][1],2)
    str_result=f'Trong tương lai gần, bạn có nguy cơ bị đột quỵ xấp xỉ bằng: {result2}%'
    df_gen_bmi=compute_bmi(df)
    df_temp=preprocess_2(df)
    # Line plot for carrier delay
    carrier_fig =  px.line(df_gen_bmi, title='Tỉ lệ đột quỵ trung bình của từng giới tính')
    # Line plot for sunbrust delay
    ss_fig=px.sunburst(df_temp, path=['work_type','smoking_status'], values='number',color='stroke'
                                ,title='Tỉ lệ đột quỵ trung bình của từng tình trạng hút thuốc ở mỗi nhóm công việc')
          
    return[str_result,carrier_fig,ss_fig]


# Run the app
if __name__ == '__main__':
    app.run_server()