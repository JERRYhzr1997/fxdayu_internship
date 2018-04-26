import pandas as pd
from jaqs_fxdayu.util import dp
from jaqs.data.dataapi import DataApi
import numpy as np
import talib as ta
from jaqs_fxdayu.research.signaldigger.process import neutralize


api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("13662241013", 
          'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTc2NDQzMzg5MTIiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTM2NjIyNDEwMTMifQ.sVIzI5VLqq8fbZCW6yZZW0ClaCkcZpFqpiK944AHEow'
)
start = 20130101
end = 20180101
SH_id = dp.index_cons(api, "000300.SH", start, end)
SZ_id = dp.index_cons(api, "000905.SH", start, end)
stock_symbol = list(set(SH_id.symbol)|set(SZ_id.symbol))



#读取数据，其中需要调节我们需要的因子的数量和内容
factor_list = ['oper_rev','oper_rev_ttm','oper_rev_lyr',"total_oper_rev",'less_gerl_admin_exp','AdminiExpenseRate','volume',"gainvariance120",'index',"cash_recp_sg_and_rs"]
check_factor = ','.join(factor_list)


import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs_fxdayu.data.dataservice import LocalDataService
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

dataview_folder = 'E:/data'
dv = DataView()
ds = LocalDataService(fp=dataview_folder)

dv_props = {'start_date': start, 'end_date': end, 'symbol':','.join(stock_symbol),
         'fields': check_factor,
         'freq': 1,
         "prepare_fields": True}

dv.init_from_config(dv_props, data_api=ds)
dv.prepare_data()



#获取行业情况
dv.add_field('sw1')
sw1 = dv.get_ts('sw1')
dict_classify = {'480000': '银行', '430000': '房地产', '460000': '休闲服务', '640000': '机械设备', '240000': '有色金属', '510000': '综合', '410000': '公用事业', '450000': '商业贸易', '730000': '通信', '330000': '家用电器', '720000': '传媒', '630000': '电气设备', '270000': '电子', '490000': '非银金融', '370000': '医药生物', '710000': '计算机', '280000': '汽车', '340000': '食品饮料', '220000': '化工', '210000': '采掘', '230000': '钢铁', '650000': '国防军工', '110000': '农林牧渔', '420000': '交通运输', '620000': '建筑装饰', '350000': '纺织服装', '610000': '建筑材料', '360000': '轻工制造'}
sw1_name = sw1.replace(dict_classify)
sw1_name.tail()



#因子编写
#gainvariance120
def cal_positive(df):
    return df[df>0]
pct_return = cal_positive(dv.get_ts('close').pct_change())
temp = pd.DataFrame({name:value.dropna().rolling(120).std()**2 for name,value in pct_return.iteritems()}, index=pct_return.index).fillna(method='ffill')
gainvariance120=dv.append_df(temp,'gainvariance120')
#alpha46
def mean(df,day):
    return df.rolling(window=day,center=False).mean()  
alpha46 = dv.add_formula('alpha46', 
               "(mean(close,3)+mean(close,6)+mean(close,12)+mean(close,24))/(4*close)",
               is_quarterly=False,
               add_data=True,
               register_funcs={"mean":mean})
#alpha48
def sum_fxdayu(df,day):
    return df.rolling(window=day,center=False).sum()  
alpha48 = dv.add_formula('alpha48', 
               "(-1*((Rank(((Sign((close - Delay(close, 1))) + Sign((Delay(close, 1) - Delay(close, 2)))) +Sign((Delay(close, 2) - Delay(close, 3)))))) * SUM(volume, 5)) / SUM(volume, 20))",
               is_quarterly=False,
               add_data=True,
               register_funcs={"SUM":sum_fxdayu})
#alpha85
alpha85 = dv.add_formula('alpha85', 
               "(Ts_Rank((volume / mean(volume,20)), 20) * Ts_Rank((-1 * Delta(close, 7)), 8))",
               is_quarterly=False,
               add_data=True,
               register_funcs={"mean":mean})
#VEMA12   直接计算一个dataframe 把没有开盘的数值设置为0
volume=dv.get_ts("volume")
volume=volume.dropna(axis=1)
vema12=pd.DataFrame(index=volume.index)
for col in volume.columns:
    inter=pd.DataFrame(pd.Series(ta.EMA(np.array(list(volume[col])),12)),columns=[col])
    for i in range(1215):
        vema12.loc[vema12.index[i],col]=inter.iloc[i,0]
temp=[]
test=list(vema12.columns)
for i in stock_symbol:
    if i not in test:
        temp=temp+[i]
for i in temp:
    vema12[i]=np.nan
dv.append_df(vema12,"vema12")
#informationratio60
hs300=ds.index_daily(['000300.SH'],20130101,20180101,"close")[0].pct_change()
close=dv.get_ts("close")
hs300=hs300.set_index(close.index)
hs300=hs300['close']
close=close.dropna(axis=1)
close=close.pct_change()
close=close.sub(hs300,axis=0)   #计算出了每日的超额收益,下面计算60日IC
ret=close.rolling(window=60).mean()
std=close.rolling(window=60).std()
ir60=close.copy()
for col in ir60.columns:
    ir60[col]=ret[col].div(std[col],axis=0)#计算出60日信息比率
dv.append_df(ir60,"ir60")
#adminiexpenserate
adminiexpenserate = dv.add_formula('adminiexpenserate', 
               "TTM(less_gerl_admin_exp)/TTM(total_oper_rev)",
               is_quarterly=False,
               add_data=True)
#ssctol
ssctol= dv.add_formula('ssctol', 
               "cash_recp_sg_and_rs/oper_rev",
               is_quarterly=False,
               add_data=True)



#数据预处理
id_zz500 = dp.daily_index_cons(api, "000300.SH", start, end)
id_hs300 = dp.daily_index_cons(api, "000905.SH", start, end)
columns_500 = list(set(id_zz500.columns)-set(id_hs300.columns))
id_member = pd.concat([id_zz500[columns_500],id_hs300],axis=1)
mask = ~id_member
# 定义可买卖条件——未停牌、未涨跌停
def limit_up_down():
    trade_status = dv.get_ts('trade_status').fillna(0)
    mask_sus = trade_status == 0
    # 涨停
    up_limit = dv.add_formula('up_limit', '(close - Delay(close, 1)) / Delay(close, 1) > 0.095', is_quarterly=False)
    # 跌停
    down_limit = dv.add_formula('down_limit', '(close - Delay(close, 1)) / Delay(close, 1) < -0.095', is_quarterly=False)
    can_enter = np.logical_and(up_limit < 1, ~mask_sus) # 未涨停未停牌
    can_exit = np.logical_and(down_limit < 1, ~mask_sus) # 未跌停未停牌
    return can_enter,can_exit
can_enter,can_exit = limit_up_down()
alpha_signal = ['gainvariance120','alpha46','alpha48','alpha85',"vema12","ir60",'adminiexpenserate','ssctol']
price = dv.get_ts('close_adj')
sw1 = sw1_name
enter = can_enter
exit =  can_exit
mask = mask
from jaqs_fxdayu.research.signaldigger.process import neutralize
neutralize_dict = {a: neutralize(factor_df = dv.get_ts(a), group = dv.get_ts("sw1")) for a in alpha_signal}



#分析因子所在行业特点
import matplotlib.pyplot as plt
from jaqs_fxdayu.research import SignalDigger
from jaqs_fxdayu.research.signaldigger import analysis
def cal_obj(signal, name, period, quantile):
#     price_bench = dv.data_benchmark
    obj = SignalDigger(output_folder="hs300/%s" % name,
                       output_format='pdf')
    obj.process_signal_before_analysis(signal,
                                   price=price,
                                   n_quantiles=quantile, period=period,
                                   mask=mask,
                                   group=sw1,
                                   can_enter = enter,
                                   can_exit = exit,
                                   commission = 0.0008
                                   )
    obj.create_full_report()
    return obj

def plot_pfm(signal, name, period=5, quantile=5):
    obj = cal_obj(signal, name, period, quantile)
    plt.show()
def signal_data(signal, name, period=5, quantile=5):
    obj = cal_obj(signal, name, period, quantile)
    return obj.signal_data

signals_dict = {a:signal_data(neutralize_dict[a], a, 20) for a in alpha_signal} 

ic_pn = pd.Panel({a: analysis.ic_stats(signals_dict[a]) for a in signals_dict.keys()})

alpha_performance = round(ic_pn.minor_xs('return_ic'),2)
print(alpha_performance)

alpha_IR = alpha_performance.loc["Ann. IR"]
alpha_IC = alpha_performance.loc["IC Mean"]

good_alpha = alpha_IC[(alpha_IC>=0.03) & (alpha_IR>=0.25)]

good_alpha_dict = {g: float('%.2f' % good_alpha[g]) for g in good_alpha.index}

good_alpha_dict



#查看银子行业特点（最优周期）
signal_dict = {alpha : signal_data(dv.get_ts(alpha), alpha, period=20, quantile=5) for alpha in good_alpha.index}

def ic_length(signal, days=750):
    return signal.loc[signal.index.levels[0][-days]:]

from jaqs.research.signaldigger import performance as pfm

performance_dict = {}
for alpha in good_alpha.index:
    ic = pfm.calc_signal_ic(ic_length(signal_dict[alpha]), by_group=True)
    mean_ic_by_group = pfm.mean_information_coefficient(ic, by_group=True)
    performance_dict[alpha] = round(mean_ic_by_group,2)

ic_industry = pd.Panel(performance_dict).minor_xs('ic')

High_IC_Industry = pd.DataFrame([ic_industry[ic_industry>=0.05][alpha].dropna(how='all') for alpha in good_alpha.index]).T



#输出     八个因子，好因子就剩下一个alpha46
alpha46 = pd.Series({'name':'alpha46','data': ['close'],'IC':good_alpha_dict['alpha46'],'type':'191因子','market':'ZZ800','classify':'sw1','Formula':"(mean(close,3)+mean(close,6)+mean(close,12)+mean(close,24))/(4*close)",'parameter':[],'description':'3、6、12、24日均线价格之和与四倍当日收盘价的比值，理论为均线拉动上涨','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha46'][indu]) for indu in High_IC_Industry['alpha46'].dropna().index}})
save_excel = pd.concat([globals()[name] for name in High_IC_Industry.columns],axis=1,keys=High_IC_Industry.columns).T
save_excel.to_excel('Finish_alpha.xlsx')
















