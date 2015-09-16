

    import numpy as np
    import os.path
    import pickle
    import pandas as pd
    os.getcwd()
    os.chdir('/Users/miricho/Desktop/Udacity/P5_MachineLearning/ud120-projects/final_project')


    data_dict = pickle.load(open("final_project_dataset.pkl", "r"))


    n = data_dict.values()
    features = ['name']
    for i in n[0].keys():
        if i != 'email_address':
            features.append(i)
            
    pre_df = [[key,value['salary'],value['to_messages'],value['deferral_payments'],value['total_payments'],
               value['exercised_stock_options'],value['bonus'],value['restricted_stock'],value['shared_receipt_with_poi'],
               value['restricted_stock_deferred'],value['total_stock_value'],value['expenses'],value['loan_advances'],
               value['from_messages'],value['other'],value['from_this_person_to_poi'],value['poi'],value['director_fees'],
               value['deferred_income'],value['long_term_incentive'],value['from_poi_to_this_person']] 
              for key, value in data_dict.iteritems() if key != 'TOTAL']
    df = pd.DataFrame(pre_df,columns = features)


    df.columns




    Index([u'name', u'salary', u'to_messages', u'deferral_payments', u'total_payments', u'exercised_stock_options', u'bonus', u'restricted_stock', u'shared_receipt_with_poi', u'restricted_stock_deferred', u'total_stock_value', u'expenses', u'loan_advances', u'from_messages', u'other', u'from_this_person_to_poi', u'poi', u'director_fees', u'deferred_income', u'long_term_incentive', u'from_poi_to_this_person'], dtype='object')




    df[['name','director_fees','restricted_stock_deferred','deferral_payments','poi']][df['poi']==True]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>director_fees</th>
      <th>restricted_stock_deferred</th>
      <th>deferral_payments</th>
      <th>poi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4  </th>
      <td>       HANNON KEVIN P</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>16 </th>
      <td>       COLWELL WESLEY</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>   27610</td>
      <td> True</td>
    </tr>
    <tr>
      <th>30 </th>
      <td>       RIEKER PAULA H</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>  214678</td>
      <td> True</td>
    </tr>
    <tr>
      <th>41 </th>
      <td>     KOPPER MICHAEL J</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>53 </th>
      <td>           SHELBY REX</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>60 </th>
      <td>     DELAINEY DAVID W</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>65 </th>
      <td>        LAY KENNETH L</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>  202911</td>
      <td> True</td>
    </tr>
    <tr>
      <th>76 </th>
      <td>   BOWEN JR RAYMOND M</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>82 </th>
      <td>     BELDEN TIMOTHY N</td>
      <td> NaN</td>
      <td> NaN</td>
      <td> 2144013</td>
      <td> True</td>
    </tr>
    <tr>
      <th>85 </th>
      <td>      FASTOW ANDREW S</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>87 </th>
      <td> CALGER CHRISTOPHER F</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>88 </th>
      <td>       RICE KENNETH D</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>95 </th>
      <td>   SKILLING JEFFREY K</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>124</th>
      <td>       YEAGER F SCOTT</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>125</th>
      <td>         HIRKO JOSEPH</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>   10259</td>
      <td> True</td>
    </tr>
    <tr>
      <th>134</th>
      <td>        KOENIG MARK E</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>141</th>
      <td>     CAUSEY RICHARD A</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
    <tr>
      <th>144</th>
      <td>      GLISAN JR BEN F</td>
      <td> NaN</td>
      <td> NaN</td>
      <td>     NaN</td>
      <td> True</td>
    </tr>
  </tbody>
</table>
</div>



#### Dataset Characteristics


    n = data_dict.values()
    print '##### Data Size #####'
    print '# of observations: ',len(n)
    print '# of total features: ',len(n[0].keys())
    
    print '##### POI #####'
    print '# of POI in dataset: ', df['poi'][df['poi']==True].count()
    print '# of non-POI in dataset: ', df['poi'][df['poi']==False].count()
    
    print '##### non-POI vs. POI #####'
    POI_median_comp = []
    POI_mean_comp = []
    POI_outlier = []
    for col in df.columns:
        if col != 'name' and col != 'poi':
            temp = df[[col]][(df['poi']==True)]
            poi_t_mean = temp[temp[col]!='NaN'].mean()
            poi_t_median = temp[temp[col]!='NaN'].median()
            #print 'mean of POI {}'.format(col), poi_t
    
            temp = df[[col]][(df['poi']==False)]
            poi_f_mean = temp[temp[col]!='NaN'].mean()
            poi_f_median = temp[temp[col]!='NaN'].median()
            #print 'mean of non-POI {}'.format(col), poi_f
            
            POI_mean_comp.append([col, round(abs(poi_t_mean-poi_f_mean),2)])
            POI_median_comp.append([col, round(abs(poi_t_median-poi_f_median),2)])
            if round(abs(poi_t_mean-poi_f_mean),2) > round(abs(poi_t_median-poi_f_median),2):
                POI_outlier.append([col,round(abs(poi_t_mean-poi_f_mean),2) - round(abs(poi_t_median-poi_f_median),2) ])
    
    #POI_comp.sort()
    POI_median_comp = sorted(POI_median_comp, key=lambda POI_median_comp: -POI_median_comp[1])
    print 'POI vs non-POI: absolute differences in median: (Descending)\n',POI_median_comp
    
    print '\n ### Features with lots of missing values: (over 50% missing rate)'
    missing_rate = []
    for col in df.columns:
        if col != 'poi' and col != 'name':
            #print col, 'missing: ', float(df[col][df[col] =='NaN'].count())/df[col].count()
            missing_rate.append([col,float(df[col][df[col] =='NaN'].count())/df[col].count()])
    low_missing_rate = [col[0] for col in missing_rate if col[1] < .5]
    high_missing_rate = [[col[0], col[1]] for col in missing_rate if col[1] >= .5]
    print high_missing_rate

    ##### Data Size #####
    # of observations:  146
    # of total features:  21
    ##### POI #####
    # of POI in dataset:  18
    # of non-POI in dataset:  127
    ##### non-POI vs. POI #####
    POI vs non-POI: absolute differences in median: (Descending)
    [['loan_advances', 80325000.0], ['exercised_stock_options', 2884228.0], ['total_stock_value', 1176506.5], ['long_term_incentive', 759333.0], ['total_payments', 697935.0], ['bonus', 575000.0], ['restricted_stock', 571445.5], ['deferred_income', 141216.0], ['other', 129947.0], ['deferral_payments', 57544.0], ['salary', 26947.0], ['expenses', 5847.5], ['shared_receipt_with_poi', 995.0], ['to_messages', 931.0], ['restricted_stock_deferred', nan], ['from_poi_to_this_person', 35.5], ['from_this_person_to_poi', 9.5], ['from_messages', 3.5], ['director_fees', nan]]
    
     ### Features with lots of missing values: (over 50% missing rate)
    [['deferral_payments', 0.7379310344827587], ['restricted_stock_deferred', 0.8827586206896552], ['loan_advances', 0.9793103448275862], ['director_fees', 0.8896551724137931], ['deferred_income', 0.6689655172413793], ['long_term_incentive', 0.5517241379310345]]



    low_missing_rate




    ['salary',
     'to_messages',
     'total_payments',
     'exercised_stock_options',
     'bonus',
     'restricted_stock',
     'shared_receipt_with_poi',
     'total_stock_value',
     'expenses',
     'from_messages',
     'other',
     'from_this_person_to_poi',
     'from_poi_to_this_person']



#### Outlier


    top_list = []
    for col in df.columns:
        if col != 'name':
            temp = []
            temp = [[i,df.iloc[index,0]] for index, i in enumerate(df[col]) if i != 'NaN']
            temp.sort(key=lambda temp: -temp[0])
            top_list.append([col,temp[:5]])
    print '##### Top 5 values and names for each feature ##### \n'
    top_list

    ##### Top 5 values and names for each feature ##### 
    





    [['salary',
      [[1111258, 'SKILLING JEFFREY K'],
       [1072321, 'LAY KENNETH L'],
       [1060932, 'FREVERT MARK A'],
       [655037, 'PICKERING MARK R'],
       [510364, 'WHALLEY LAWRENCE G']]],
     ['to_messages',
      [[15149, 'SHAPIRO RICHARD S'],
       [12754, 'KEAN STEVEN J'],
       [8305, 'KITCHEN LOUISE'],
       [7991, 'BELDEN TIMOTHY N'],
       [7315, 'BECK SALLY W']]],
     ['deferral_payments',
      [[6426990, 'FREVERT MARK A'],
       [3131860, 'HORTON STANLEY C'],
       [2964506, 'HUMPHREY GENE E'],
       [2869717, 'ALLEN PHILLIP K'],
       [2157527, 'HAEDICKE MARK E']]],
     ['total_payments',
      [[103559793, 'LAY KENNETH L'],
       [17252530, 'FREVERT MARK A'],
       [15456290, 'BHATNAGAR SANJAY'],
       [10425757, 'LAVORATO JOHN J'],
       [8682716, 'SKILLING JEFFREY K']]],
     ['exercised_stock_options',
      [[34348384, 'LAY KENNETH L'],
       [30766064, 'HIRKO JOSEPH'],
       [19794175, 'RICE KENNETH D'],
       [19250000, 'SKILLING JEFFREY K'],
       [15364167, 'PAI LOU L']]],
     ['bonus',
      [[8000000, 'LAVORATO JOHN J'],
       [7000000, 'LAY KENNETH L'],
       [5600000, 'SKILLING JEFFREY K'],
       [5249999, 'BELDEN TIMOTHY N'],
       [4175000, 'ALLEN PHILLIP K']]],
     ['restricted_stock',
      [[14761694, 'LAY KENNETH L'],
       [13847074, 'WHITE JR THOMAS E'],
       [8453763, 'PAI LOU L'],
       [6843672, 'SKILLING JEFFREY K'],
       [4188667, 'FREVERT MARK A']]],
     ['shared_receipt_with_poi',
      [[5521, 'BELDEN TIMOTHY N'],
       [4527, 'SHAPIRO RICHARD S'],
       [3962, 'LAVORATO JOHN J'],
       [3920, 'WHALLEY LAWRENCE G'],
       [3669, 'KITCHEN LOUISE']]],
     ['restricted_stock_deferred',
      [[15456290, 'BHATNAGAR SANJAY'],
       [44093, 'BELFER ROBERT'],
       [-32460, 'CHAN RONNIE'],
       [-44093, 'JAEDICKE ROBERT'],
       [-72419, 'GATHMANN WILLIAM D']]],
     ['total_stock_value',
      [[49110078, 'LAY KENNETH L'],
       [30766064, 'HIRKO JOSEPH'],
       [26093672, 'SKILLING JEFFREY K'],
       [23817930, 'PAI LOU L'],
       [22542539, 'RICE KENNETH D']]],
     ['expenses',
      [[228763, 'MCCLELLAN GEORGE'],
       [228656, 'URQUHART JOHN A'],
       [178979, 'SHANKMAN JEFFREY A'],
       [137767, 'SHAPIRO RICHARD S'],
       [137108, 'MCMAHON JEFFREY']]],
     ['loan_advances',
      [[81525000, 'LAY KENNETH L'],
       [2000000, 'FREVERT MARK A'],
       [400000, 'PICKERING MARK R']]],
     ['from_messages',
      [[14368, 'KAMINSKI WINCENTY J'],
       [6759, 'KEAN STEVEN J'],
       [4343, 'BECK SALLY W'],
       [3069, 'DELAINEY DAVID W'],
       [2742, 'MCCONNELL MICHAEL S']]],
     ['other',
      [[10359729, 'LAY KENNETH L'],
       [7427621, 'FREVERT MARK A'],
       [2818454, 'MARTIN AMANDA K'],
       [2660303, 'BAXTER JOHN C'],
       [1852186, 'SHERRIFF JOHN R']]],
     ['from_this_person_to_poi',
      [[609, 'DELAINEY DAVID W'],
       [411, 'LAVORATO JOHN J'],
       [387, 'KEAN STEVEN J'],
       [386, 'BECK SALLY W'],
       [194, 'KITCHEN LOUISE']]],
     ['poi',
      [[True, 'HANNON KEVIN P'],
       [True, 'COLWELL WESLEY'],
       [True, 'RIEKER PAULA H'],
       [True, 'KOPPER MICHAEL J'],
       [True, 'SHELBY REX']]],
     ['director_fees',
      [[137864, 'BHATNAGAR SANJAY'],
       [125034, 'SAVAGE FRANK'],
       [119292, 'GRAMM WENDY L'],
       [113784, 'BLAKE JR. NORMAN P'],
       [112492, 'LEMAISTRE CHARLES']]],
     ['deferred_income',
      [[-833, 'BOWEN JR RAYMOND M'],
       [-1042, 'GAHN ROBERT S'],
       [-4167, 'SHELBY REX'],
       [-5104, 'BANNANTINE JAMES M'],
       [-10800, 'WESTFAHL RICHARD K']]],
     ['long_term_incentive',
      [[5145434, 'MARTIN AMANDA K'],
       [3600000, 'LAY KENNETH L'],
       [2234774, 'ECHOLS JOHN B'],
       [2035380, 'LAVORATO JOHN J'],
       [1920000, 'SKILLING JEFFREY K']]],
     ['from_poi_to_this_person',
      [[528, 'LAVORATO JOHN J'],
       [305, 'DIETRICH JANET R'],
       [251, 'KITCHEN LOUISE'],
       [242, 'FREVERT MARK A'],
       [240, 'COLWELL WESLEY']]]]


