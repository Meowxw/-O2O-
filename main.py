import pandas as pd
import numpy as np
from datetime import date
from datetime import *

starttime = datetime.now()
print('开始运行时间：{}'.format(starttime.strftime('%Y-%m-%d %H:%M:%S')))

############# data_set split ##################3
print('开始读取数据')
#1754884 record,1053282 with coupon_id,9738 coupon. date_received:20160101~20160615,date:20160101~20160630, 539438 users, 8415 merchants
off_train = pd.read_csv('data/ccf_offline_stage1_train.csv')
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
#2050 coupon_id. date_received:20160701~20160731, 76309 users(76307 in trainset, 35965 in online_trainset), 1559 merchants(1558 in trainset)
off_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
#11429826 record(872357 with coupon_id),762858 user(267448 in off_train)
on_train = pd.read_csv('data/ccf_online_stage1_train.csv')
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']
print('完成读取数据')

#shujujihuafen
print('开始数据划分')
dataset3 = off_test
feature3 = off_train[(off_train.date_received>='20160315')&(off_train.date_received<='20160630')]
dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]
feature2 = off_train[(off_train.date_received>='20160201')&(off_train.date_received<='20160514')]
dataset1 = off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')]
feature1 = off_train[(off_train.date_received>='20160101')&(off_train.date_received<='20160413')]

print('完成数据划分')


#online
print('线上读取数据')
on_dataset3 = off_test
on_feature3 = on_train[(on_train.date_received>='20160315')&(on_train.date_received<='20160630')]
on_dataset2 = on_train[(on_train.date_received>='20160515')&(on_train.date_received<='20160615')]
on_feature2 = on_train[(on_train.date_received>='20160201')&(on_train.date_received<='20160514')]
on_dataset1 = on_train[(on_train.date_received>='20160414')&(on_train.date_received<='20160514')]
on_feature1 = on_train[(on_train.date_received>='20160101')&(on_train.date_received<='20160413')]
print('完成线上读取数据')

def genFeature(dataset1,on_dataset1,label_dataset1):               
    ############# 1.user offline:  ##################3
    #用户领取优惠券次数,['user_id', 'coupon_id', '1-1_2', '1-1_1']
    t11_1=dataset1[['user_id']]
    t11_1['1-1_1']=1
    t11_1=t11_1.groupby(['user_id']).agg('sum').reset_index()
    
    t11_2=dataset1[['user_id','coupon_id']]
    t11_2['1-1_2']=1
    t11_2=t11_2.groupby(['user_id','coupon_id']).agg('sum').reset_index()
    
    t11=pd.merge(t11_2,t11_1,on=['user_id'],how='left')
    
    #用户获得优惠券但没有消费的次数,['user_id', 'coupon_id', '1-2_2', '1-2_1']
    t12_1=dataset1[(dataset1.coupon_id!='null')&(dataset1.date=='null')][['user_id','coupon_id']]
    t12_1['1-2_1']=1
    t12_1=t12_1.groupby(['user_id']).agg('sum').reset_index()
    
    t12_2=dataset1[(dataset1.coupon_id!='null')&(dataset1.date=='null')][['user_id','coupon_id']]
    t12_2['1-2_2']=1
    t12_2=t12_2.groupby(['user_id','coupon_id']).agg('sum').reset_index()
    
    t12=pd.merge(t12_2,t12_1,on=['user_id'],how='left')
    
    #用户获得优惠券并核销次数,['user_id', 'coupon_id', '1-3_2', '1-3_1']
    t13_1=dataset1[(dataset1.coupon_id!='null')&(dataset1.date!='null')][['user_id','coupon_id']]
    t13_1['1-3_1']=1
    t13_1=t13_1.groupby(['user_id']).agg('sum').reset_index()
    
    t13_2=dataset1[(dataset1.coupon_id!='null')&(dataset1.date!='null')][['user_id','coupon_id']]
    t13_2['1-3_2']=1
    t13_2=t13_2.groupby(['user_id','coupon_id']).agg('sum').reset_index()
    
    t13=pd.merge(t13_2,t13_1,on=['user_id'],how='left')
    
    #用户领取优惠券后进行核销率,['user_id', 'coupon_id', '1-4_2', '1-4_1']
    t14_1 = pd.merge(t11_1,t13_1,on=['user_id'],how='left')
    t14_1=t14_1.fillna(0)
    t14_1['1-4_1']=t14_1['1-3_1']/t14_1['1-1_1']
    
    t14_2 = pd.merge(t11_2,t13_2,on=['user_id','coupon_id'],how='left')
    t14_2=t14_2.fillna(0)
    t14_2['1-4_2']=t14_2['1-3_2']/t14_2['1-1_2']
    
    t14=pd.merge(t14_2,t14_1,on=['user_id'],how='left')
    t14=t14[['user_id','coupon_id','1-4_2','1-4_1']]
    
    #用户满0200/200~500 减的优惠券核销率,['get_discount_man', 'get_discount_jian', '15_1', '15_2']
    def calc_discount_rate(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return float(s[0])
        else:
            return 1.0-float(s[1])/float(s[0])
    
    def get_discount_man(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return 'null'
        else:
            return int(s[0])
            
    def get_discount_jian(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return 'null'
        else:
            return int(s[1])
            
    def is_man_jian(s):
        s=str(s)
        s=s.split(':')
        if len(s)==1:
            return 0
        else:
            return 1
    
    #man_set
    #Out[46]: {100, 'null', 5, 200, 10, 300, 50, 20, 150, 30}
    #jian_set
    #Out[47]: [1, 'null', 5, 10, 50, 20, 30]
    #不同优惠券核销率
    t15=dataset1[['date','coupon_id','discount_rate']]
    #t15['calc_discount_rate']=dataset1.discount_rate.apply(calc_discount_rate)
    t15['get_discount_man']=dataset1.discount_rate.apply(get_discount_man)
    t15['get_discount_jian']=dataset1.discount_rate.apply(get_discount_jian)
    #t15['is_man_jian']=dataset1.discount_rate.apply(is_man_jian)
    #man_set=list(set(t15['get_discount_man']))
    #jian_set=list(set(t15['get_discount_jian']))
    
    t15_1=t15[(t15.date!='null')&(t15.coupon_id!='null')][['get_discount_man']]
    t15_1['man_he']=1
    t15_1=t15_1.groupby('get_discount_man').agg('sum').reset_index()
    t15_111=t15[['get_discount_man']]
    t15_111['man_sum']=1
    t15_11=t15_111.groupby('get_discount_man').agg('sum').reset_index()
    t15_1=pd.merge(t15_11,t15_1,on='get_discount_man',how='left')
    t15_1=t15_1.fillna('0')
    t15_1['15_1']=np.int64(t15_1['man_he'])/t15_1['man_sum']
    t15_1=t15_1[['get_discount_man','15_1']]
    
    t15_2=t15[(t15.date!='null')&(t15.coupon_id!='null')][['get_discount_jian']]
    t15_2['jian_he']=1
    t15_2=t15_2.groupby('get_discount_jian').agg('sum').reset_index()
    t15_222=t15[['get_discount_jian']]
    t15_222['jian_sum']=1
    t15_22=t15_222.groupby('get_discount_jian').agg('sum').reset_index()
    t15_2=pd.merge(t15_22,t15_2,on='get_discount_jian',how='left')
    t15_2=t15_2.fillna('0')
    t15_2['15_2']=np.int64(t15_2['jian_he'])/t15_2['jian_sum']
    t15_2=t15_2[['get_discount_jian','15_2']]
    
    #t15=pd.merge(t15,t15_1,on=['get_discount_man'],how='left')
    #t15=pd.merge(t15,t15_2,on=['get_discount_jian'],how='left')
    #t15=t15[['get_discount_man','get_discount_jian','15_1','15_2']]
    
    ##用户核销满0200/200~500减的优惠券占顾客核销优惠券的比重
    #不同额度优惠券满或减占顾客总优惠券比重,['user_id', 'get_discount_man', 'get_discount_jian', '16_1', '16_2']
    t16=dataset1[['user_id','date','coupon_id','discount_rate']]
    t16['get_discount_man']=dataset1.discount_rate.apply(get_discount_man)
    t16['get_discount_jian']=dataset1.discount_rate.apply(get_discount_jian)
    
    t16_1=t16[(t16.date!='null')&(t16.coupon_id!='null')][['user_id','get_discount_man']]
    t16_1['man_he']=1
    t16_1=t16_1.groupby(['user_id','get_discount_man']).agg('sum').reset_index()
    t16_111=t16[['user_id','get_discount_man']]
    t16_111['man_sum']=1
    t16_11=t16_111.groupby(['user_id','get_discount_man']).agg('sum').reset_index()
    t16_1=pd.merge(t16_11,t16_1,on=['user_id','get_discount_man'],how='left')
    t16_1=t16_1.fillna('0')
    t16_1['16_1']=np.int64(t16_1['man_he'])/t16_1['man_sum']
    t16_1=t16_1[['user_id','get_discount_man','16_1']]
    
    t16_2=t16[(t16.date!='null')&(t16.coupon_id!='null')][['user_id','get_discount_jian']]
    t16_2['jian_he']=1
    t16_2=t16_2.groupby(['user_id','get_discount_jian']).agg('sum').reset_index()
    t16_222=t16[['user_id','get_discount_jian']]
    t16_222['jian_sum']=1
    t16_22=t16_222.groupby(['user_id','get_discount_jian']).agg('sum').reset_index()
    t16_2=pd.merge(t16_22,t16_2,on=['user_id','get_discount_jian'],how='left')
    t16_2=t16_2.fillna('0')
    t16_2['16_2']=np.int64(t16_2['jian_he'])/t16_2['jian_sum']
    t16_2=t16_2[['user_id','get_discount_jian','16_2']]
    
    #t16=pd.merge(t16,t16_1,on=['user_id','get_discount_man'],how='left')
    #t16=pd.merge(t16,t16_2,on=['user_id','get_discount_jian'],how='left')
    #t16=t16[['user_id','get_discount_man','get_discount_jian','16_1','16_2']]
    
    ##用户核销优惠券的平均/中位数/最低/最高消费折率,['user_id', '17_1mean', '17_2median', '17_3min', '17_4max']
    t17=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id','discount_rate']]
    t17['calc_discount_rate']=dataset1.discount_rate.apply(calc_discount_rate)
    t17=t17[['user_id','calc_discount_rate']]
    t17_1=t17.groupby(['user_id']).agg('mean').reset_index()
    t17_1.rename(columns={'calc_discount_rate':'17_1mean'},inplace=True)
    t17_2=t17.groupby(['user_id']).agg('median').reset_index()
    t17_2.rename(columns={'calc_discount_rate':'17_2median'},inplace=True)
    t17_3=t17.groupby(['user_id']).agg('min').reset_index()
    t17_3.rename(columns={'calc_discount_rate':'17_3min'},inplace=True)
    t17_4=t17.groupby(['user_id']).agg('max').reset_index()
    t17_4.rename(columns={'calc_discount_rate':'17_4max'},inplace=True)
    t17=pd.merge(t17_1,t17_2,on='user_id',how='left')
    t17=pd.merge(t17,t17_3,on='user_id',how='left')
    t17=pd.merge(t17,t17_4,on='user_id',how='left')
    
    ##用户核销过优惠券的商家编号(one hot),['user_id', 'merchant_id_17', 'merchant_id_18', 'merchant_id_20',
    #       'merchant_id_25', 'merchant_id_28', 'merchant_id_32', 'merchant_id_33',
    #       'merchant_id_35', 'merchant_id_40',
    #       ...
    #       'merchant_id_8773', 'merchant_id_8775', 'merchant_id_8786',
    #       'merchant_id_8807', 'merchant_id_8825', 'merchant_id_8828',
    #       'merchant_id_8831', 'merchant_id_8839', 'merchant_id_8843',
    #       'merchant_id_8844']
    #t181=t18.groupby(['user_id','merchant_id']).agg('count').reset_index()
    #(t181!=1).sum()
    #
    #Out[32]: 
    #user_id        7880
    #merchant_id    7880
    #count             0
    #dtype: int64
    #t18=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id','merchant_id']]
    #t18= pd.get_dummies(t18, columns=['merchant_id'])
    #t18=t18.groupby('user_id').agg('sum').reset_index()
    
    ##用户核销过的不同优惠券数量，及其占所有不同优惠券的比重,['user_id', 'coupon_id_10', 'coupon_id_100', 'coupon_id_10009',
    #       'coupon_id_10012', 'coupon_id_10013', 'coupon_id_10017',
    #       'coupon_id_1002', 'coupon_id_10029', 'coupon_id_10030',
    #       ...
    #       'coupon_id_9948', 'coupon_id_995', 'coupon_id_9958', 'coupon_id_9962',
    #       'coupon_id_9971', 'coupon_id_9972', 'coupon_id_9986', 'coupon_id_9987',
    #       'coupon_id_9988', 'coupon_id_9997']
    #t19=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id','coupon_id']]
    #t19= pd.get_dummies(t19, columns=['coupon_id'])
    #t19=t19.groupby('user_id').agg('sum').reset_index()
    
    #用户核销每个商家多少张优惠券,['user_id', 'merchant_id', '110_1', '110_2', '110_3', '110_4']
    t110=dataset1[(dataset1.coupon_id!='null')&(dataset1.date!='null')][['user_id','merchant_id']]
    t110['110']=1
    t110_1=t110.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    t110_1.rename(columns={'110':'110_1'},inplace=True)
    t110_2=t110.groupby(['user_id','merchant_id']).agg('mean').reset_index()
    t110_2.rename(columns={'110':'110_2'},inplace=True)
    t110_3=t110.groupby(['user_id']).agg('sum').reset_index()
    t110_3.rename(columns={'110':'110_3'},inplace=True)
    t110_3=t110_3[['user_id','110_3']]
    t110_4=t110.groupby(['user_id']).agg('mean').reset_index()
    t110_4.rename(columns={'110':'110_4'},inplace=True)
    t110_4=t110_4[['user_id','110_4']]
    t110=pd.merge(t110_1,t110_2,on=['user_id','merchant_id'],how='left')
    t110=pd.merge(t110,t110_3,on=['user_id'],how='left')
    t110=pd.merge(t110,t110_4,on=['user_id'],how='left')
    
    #用户核销优惠券中的平均/最大/最小用户-商家距离,['user_id', 'min_distance', 'max_distance', 'mean_distance',
    #       'median_distance']
    t111=dataset1[(dataset1.coupon_id!='null')&(dataset1.date!='null')][['user_id','distance']]
    t111.replace('null',-1,inplace=True)
    t111.distance = t111.distance.astype('int')
    t111.replace(-1,np.nan,inplace=True)
    t111_1 = t111.groupby('user_id').agg('min').reset_index()
    t111_1.rename(columns={'distance':'min_distance'},inplace=True)
    t111_2 = t111.groupby('user_id').agg('max').reset_index()
    t111_2.rename(columns={'distance':'max_distance'},inplace=True)
    t111_3 = t111.groupby('user_id').agg('mean').reset_index()
    t111_3.rename(columns={'distance':'mean_distance'},inplace=True)
    t111_4 = t111.groupby('user_id').agg('median').reset_index()
    t111_4.rename(columns={'distance':'median_distance'},inplace=True)
    t111 = pd.merge(t111_1,t111_2,on=['user_id'],how='left')
    t111 = pd.merge(t111,t111_3,on=['user_id'],how='left')
    t111 = pd.merge(t111,t111_4,on=['user_id'],how='left')
    
    #特征一：user offline:
    dataset1['get_discount_man']=dataset1.discount_rate.apply(get_discount_man)
    dataset1['get_discount_jian']=dataset1.discount_rate.apply(get_discount_jian)
    feature_1=dataset1[['user_id','coupon_id','get_discount_man','get_discount_jian','merchant_id']]
    feature_1=pd.merge(feature_1,t11,on=['user_id','coupon_id'],how='left')
    feature_1=pd.merge(feature_1,t12,on=['user_id','coupon_id'],how='left')
    feature_1=pd.merge(feature_1,t13,on=['user_id','coupon_id'],how='left')
    feature_1=pd.merge(feature_1,t14,on=['user_id','coupon_id'],how='left')
    feature_1=pd.merge(feature_1,t15_1,on=['get_discount_man'],how='left')
    feature_1=pd.merge(feature_1,t15_2,on=['get_discount_jian'],how='left')
    feature_1=pd.merge(feature_1,t16_1,on=['user_id','get_discount_man'],how='left')
    feature_1=pd.merge(feature_1,t16_2,on=['user_id','get_discount_jian'],how='left')
    feature_1=pd.merge(feature_1,t17,on=['user_id'],how='left')
    #feature_1=pd.merge(feature_1,t18,on=['user_id'],how='left')
    #feature_1=pd.merge(feature_1,t19,on=['user_id'],how='left')
    feature_1=pd.merge(feature_1,t110,on=['user_id','merchant_id'],how='left')
    feature_1=pd.merge(feature_1,t111,on=['user_id'],how='left')
    print (feature_1.shape)
    
    del t11,t11_1,t11_2,t12,t12_1,t12_2,t13,t13_1,t13_2,t14,t14_1,t14_2,\
        t15,t15_1,t15_11,t15_111,t15_2,t15_22,t15_222,t16,t16_1,t16_11,\
        t16_111,t16_2,t16_22,t16_222,t17,t17_1,t17_2,t17_3,t17_4,\
        t110,t110_1,t110_2,t110_3,t110_4,t111,t111_1,t111_2,t111_3,t111_4
    
    feature_1.drop_duplicates(inplace=True)
        
    ############# 2.user online:  ##################3
    #用户线上操作次数,user_id
    t21=on_dataset1[['user_id']]
    t21['2-1']=1
    t21=t21.groupby('user_id').agg('sum').reset_index()
    
    #用户线上点击次数、点击率,user_id
    t22=on_dataset1[(on_dataset1.action==0)][['user_id']]
    t22['2-2_1']=1
    t22_1=t22.groupby('user_id').agg('sum').reset_index()
    t22=pd.merge(t21,t22_1,on=['user_id'],how='left')
    t22=t22.fillna(0)
    t22['2-2_2']=t22['2-2_1']/t22['2-1']
    t22=t22[['user_id','2-2_1','2-2_2']]
    
    #用户线上购买次数、购买率、购买/点击比,user_id
    def correctnan(s):
        if s==0:
            return -1
    def returnfuyi(s):
        if s<=0:
            return 0
    t23=on_dataset1[(on_dataset1.action==1)][['user_id']]
    t23['2-3_1']=1
    t23_1=t23.groupby('user_id').agg('sum').reset_index()
    t23=pd.merge(t21,t23_1,on=['user_id'],how='left')
    t23=pd.merge(t23,t22,on=['user_id'],how='left')
    t23=t23[['user_id','2-1','2-3_1','2-2_1']]
    t23=t23.fillna(0)
    t23['2-3_2']=t23['2-3_1']/t23['2-1']
    t23['2-2_1']=t23['2-2_1'].apply(correctnan)
    t23['2-3_3']=t23['2-3_1']/t23['2-2_1']
    t23['2-3_3']=t23['2-3_3'].apply(returnfuyi)
    t23=t23[['user_id','2-3_1','2-3_2','2-3_3']]
    
    #用户线上仅领取次数、领取率、领取/点击比,user_id
    t24=on_dataset1[(on_dataset1.action==2)][['user_id']]
    t24['2-4_1']=1
    t24_1=t24.groupby('user_id').agg('sum').reset_index()
    t24=pd.merge(t21,t24_1,on=['user_id'],how='left')
    t24=pd.merge(t24,t22,on=['user_id'],how='left')
    t24=t24[['user_id','2-1','2-4_1','2-2_1']]
    t24=t24.fillna(0)
    t24['2-4_2']=t24['2-4_1']/t24['2-1']
    t24['2-2_1']=t24['2-2_1'].apply(correctnan)
    t24['2-4_3']=t24['2-4_1']/t24['2-2_1']
    t24['2-4_3']=t24['2-4_3'].apply(returnfuyi)
    t24=t24[['user_id','2-4_1','2-4_2','2-4_3']]
    
    #用户线上不消费次数,user_id
    t25=on_dataset1[on_dataset1.date=='null'][['user_id']]
    t25['2-5']=1
    t25=t25.groupby('user_id').agg('sum').reset_index()
    
    #用户线上优惠券核销次数,user_id 
    t26=on_dataset1[(on_dataset1.date!='null')&(on_dataset1.coupon_id!='null')][['user_id']]
    t26['2-6']=1
    t26=t26.groupby('user_id').agg('sum').reset_index()
    
    #用户线上优惠券核销率,user_id
    t27=pd.merge(t21,t26,on=['user_id'],how='left')
    t27=t27.fillna(0)
    t27['2-7']=t27['2-6']/t27['2-1']
    t27=t27[['user_id','2-7']]
    
    #用户线下不消费次数占线上线下总的不消费次数的比重,user_id
    t28=dataset1[dataset1.date=='null'][['user_id']]
    t28['2-8']=1
    t28=t28.groupby('user_id').agg('sum').reset_index()
    t28=pd.merge(t25,t28,on=['user_id'],how='outer')
    t28=t28.fillna(0)
    t28['2-8']=t28['2-8']/(t28['2-5']+t28['2-8'])
    t28['user_id']=np.int64(t28['user_id'])
    t28=t28[['user_id','2-8']]
    
    #用户线上消费次数,user_id
    t29=on_dataset1[on_dataset1.date!='null'][['user_id']]
    t29['2-9']=1
    t29=t29.groupby('user_id').agg('sum').reset_index()
    
    #用户线下消费次数占线上线下总的消费次数的比重, user_id 
    t210=dataset1[dataset1.date!='null'][['user_id']]
    t210['2-10']=1
    t210=t210.groupby('user_id').agg('sum').reset_index()
    t210=pd.merge(t210,t29,on=['user_id'],how='outer')
    t210=t210.fillna(0)
    t210['2-10']=t210['2-10']/(t210['2-10']+t210['2-9'])
    t210['user_id']=np.int64(t210['user_id'])
    t210=t210[['user_id','2-10']]
    
    #用户线下的优惠券核销次数占线上线下总的优惠券核销次数的比重,user_id
    t211=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id']]
    t211['2-11']=1
    t211=t211.groupby('user_id').agg('sum').reset_index()
    t211=pd.merge(t211,t26,on=['user_id'],how='outer')
    t211=t211.fillna(0)
    t211['2-11']=t211['2-6']/(t211['2-11']+t211['2-6'])
    t211['user_id']=np.int64(t211['user_id'])
    t211=t211[['user_id','2-11']]
    
    #用户线上领取次数、领取率、领取/点击比,user_id
    t212=on_dataset1[(on_dataset1.date_received!='null')][['user_id']]
    t212['2-12_1']=1
    t212=t212.groupby('user_id').agg('sum').reset_index()
    t212=pd.merge(t21,t212,on=['user_id'],how='left')
    t212=pd.merge(t212,t22,on=['user_id'],how='left')
    t212=t212[['user_id','2-1','2-12_1','2-2_1']]
    t212=t212.fillna(0)
    t212['2-12_2']=t212['2-12_1']/t212['2-1']
    t212['2-2_1']=t212['2-2_1'].apply(correctnan)
    t212['2-12_3']=t212['2-12_1']/t212['2-2_1']
    t212['2-12_3']=t212['2-12_3'].apply(returnfuyi)
    t212=t212[['user_id','2-12_1','2-12_2','2-12_3']]
    
    #用户线下领取的记录数量占总的记录数量的比重,user_id
    t213=dataset1[(dataset1.date_received!='null')][['user_id']]
    t213['2-13_1']=1
    t213=t213.groupby('user_id').agg('sum').reset_index()
    t213=pd.merge(t213,t212,on=['user_id'],how='outer')
    t213=t213[['user_id','2-13_1','2-12_1']]
    t213=t213.fillna(0)
    t213['2-13']=t213['2-13_1']/(t213['2-13_1']+t213['2-12_1'])
    t213['user_id']=np.int64(t213['user_id'])
    t213=t213[['user_id','2-13']]
    
    #特征二：user online：
    feature_2=on_dataset1[['user_id']]
    feature_2.drop_duplicates(inplace=True)
    feature_2=pd.merge(feature_2,t21,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t22,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t23,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t24,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t25,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t26,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t27,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t28,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t29,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t210,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t211,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t212,on=['user_id'],how='left')
    feature_2=pd.merge(feature_2,t213,on=['user_id'],how='left')
    print (feature_2.shape)
    
    del t21,t210,t211,t212,t213,t22,t22_1,t23,t23_1,t24,t24_1,t25,t26,t27,t28,t29
    
    feature_2.drop_duplicates(inplace=True)
    
    ############# 3.merchant related:  ##################3
    #商家优惠券被领取次数,merchant_id  
    t31=dataset1[['merchant_id']]
    t31['3-1']=1
    t31=t31.groupby('merchant_id').agg('sum').reset_index()
    
    #商家优惠券被领取后不核销次数,merchant_id  
    t32=dataset1[(dataset1.date=='null')&(dataset1.coupon_id!='null')][['merchant_id']]
    t32['3-2']=1
    t32=t32.groupby('merchant_id').agg('sum').reset_index()
    
    #商家优惠券被领取后核销次数,merchant_id  
    t33=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id']]
    t33['3-3']=1
    t33=t33.groupby('merchant_id').agg('sum').reset_index()
    
    #商家优惠券被领取后核销率,merchant_id  
    t34=pd.merge(t31,t33,on=['merchant_id'],how='left')
    t34=t34.fillna(0)
    t34['3-4']=t34['3-3']/t34['3-1']
    t34=t34[['merchant_id','3-4']]
    
    #商家优惠券核销的中位数/平均/最小/最大消费折率,merchant_id
    t35=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id','discount_rate']]
    t35['calc_discount_rate']=t35.discount_rate.apply(calc_discount_rate)
    t35_1=t35.groupby('merchant_id').agg('mean').reset_index()
    t35_1.rename(columns={'calc_discount_rate':'mean_discount'},inplace=True)
    t35_2=t35.groupby('merchant_id').agg('median').reset_index()
    t35_2.rename(columns={'calc_discount_rate':'median_discount'},inplace=True)
    t35_3=t35.groupby('merchant_id').agg('max').reset_index()
    t35_3.rename(columns={'calc_discount_rate':'max_discount'},inplace=True)
    t35_3=t35_3[['merchant_id','max_discount']]
    t35_4=t35.groupby('merchant_id').agg('min').reset_index()
    t35_4.rename(columns={'calc_discount_rate':'min_discount'},inplace=True)
    t35_4=t35_4[['merchant_id','min_discount']]
    t35=pd.merge(t35_1,t35_2,on=['merchant_id'],how='left')
    t35=pd.merge(t35,t35_3,on=['merchant_id'],how='left')
    t35=pd.merge(t35,t35_4,on=['merchant_id'],how='left')
    
    #商家销量,merchant_id
    t36=dataset1[(dataset1.date!='null')][['merchant_id']]
    t36['3-6']=1
    t36=t36.groupby('merchant_id').agg('sum').reset_index()
    
    #核销商家优惠券的不同用户数量，及其占领取不同的用户比重,user_id  merchant_id
    t37=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id','merchant_id']]
    t37['3-7']=1
    t37=t37.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    
    #商家优惠券平均每个用户核销多少张,user_id  merchant_id
    t38=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id','merchant_id']]
    t38['3-8']=1
    t38=t38.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    t38_1=t38[['user_id','3-8']]
    t38_1=t38_1.groupby(['user_id']).agg('mean').reset_index()
    t38_2=t38[['merchant_id','3-8']]
    t38_2=t38_2.groupby(['merchant_id']).agg('mean').reset_index()
    t38=t38[['user_id','merchant_id']]
    t38=pd.merge(t38,t38_1,on=['user_id'],how='left')
    t38=pd.merge(t38,t38_2,on=['merchant_id'],how='left')
    t38.rename(columns={'3-8_x':'3-8_1','3-8_y':'3-8_2'},inplace=True)
    
    #商家被核销过的不同优惠券数量,merchant_id coupon_id
    t39=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id','coupon_id']]
    t39['3-9']=1
    t39=t39.groupby(['merchant_id','coupon_id']).agg('sum').reset_index()
    
    #商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重,merchant_id coupon_id
    t310_1=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id','coupon_id']]
    t310_1['3-10_1']=1
    t310_1=t310_1.groupby('merchant_id').agg('sum').reset_index()
    t310_2=dataset1[['merchant_id','coupon_id']]
    t310_2['3-10_2']=1
    t310_2=t310_2.groupby('merchant_id').agg('sum').reset_index()
    t310=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id','coupon_id']]
    t310.drop_duplicates(inplace=True)
    t310=pd.merge(t310,t310_2,on=['merchant_id'],how='left')
    t310=pd.merge(t310,t310_1,on=['merchant_id'],how='left')
    t310['3-10']=t310['3-10_1']/t310['3-10_2']
    t310=t310[['merchant_id','coupon_id','3-10']]
    
    #商家平均每种优惠券核销多少张,merchant_id coupon_id
    t311=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id','coupon_id']]
    t311['3-11']=1
    t311=t311.groupby(['merchant_id','coupon_id']).agg('sum').reset_index()
    t311_1=t311[['merchant_id','3-11']]
    t311_1=t311_1.groupby(['merchant_id']).agg('mean').reset_index()
    t311_2=t311[['coupon_id','3-11']]
    t311_2=t311_2.groupby(['coupon_id']).agg('mean').reset_index()
    t311=t311[['merchant_id','coupon_id']]
    t311=pd.merge(t311,t311_1,on=['merchant_id'],how='left')
    t311=pd.merge(t311,t311_2,on=['coupon_id'],how='left')
    t311.rename(columns={'3-11_x':'3-11_1','3-11_y':'3-11_2'},inplace=True)
    
    #商家被核销优惠券的平均时间率,merchant_id
    t312=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id']]
    t312.drop_duplicates(inplace=True)
    
    #商家被核销优惠券中的平均/最小/最大用户-商家距离,merchant_id
    t313=dataset1[(dataset1.coupon_id!='null')&(dataset1.date!='null')][['merchant_id','distance']]
    t313.replace('null',-1,inplace=True)
    t313.distance = t313.distance.astype('int')
    t313.replace(-1,np.nan,inplace=True)
    t313_1 = t313.groupby('merchant_id').agg('min').reset_index()
    t313_1.rename(columns={'distance':'min_distance'},inplace=True)
    t313_2 = t313.groupby('merchant_id').agg('max').reset_index()
    t313_2.rename(columns={'distance':'max_distance'},inplace=True)
    t313_3 = t313.groupby('merchant_id').agg('mean').reset_index()
    t313_3.rename(columns={'distance':'mean_distance'},inplace=True)
    t313_4 = t313.groupby('merchant_id').agg('median').reset_index()
    t313_4.rename(columns={'distance':'median_distance'},inplace=True)
    t313 = pd.merge(t313_1,t313_2,on=['merchant_id'],how='left')
    t313 = pd.merge(t313,t313_3,on=['merchant_id'],how='left')
    t313 = pd.merge(t313,t313_4,on=['merchant_id'],how='left')
    
    #商家最欢迎优惠券
    #商家最欢迎用户
    #特征三：merchant related:
    feature_3=dataset1[['user_id','merchant_id','coupon_id']]
    feature_3=pd.merge(feature_3,t31,on=['merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t32,on=['merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t33,on=['merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t34,on=['merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t35,on=['merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t36,on=['merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t37,on=['user_id','merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t38,on=['user_id','merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t39,on=['merchant_id','coupon_id'],how='left')
    feature_3=pd.merge(feature_3,t310,on=['merchant_id','coupon_id'],how='left')
    feature_3=pd.merge(feature_3,t311,on=['merchant_id','coupon_id'],how='left')
    feature_3=pd.merge(feature_3,t312,on=['merchant_id'],how='left')
    feature_3=pd.merge(feature_3,t313,on=['merchant_id'],how='left')
    print (feature_3.shape)
    
    del t31,t310,t310_1,t310_2,t311,t311_1,t311_2,t312,t313,t313_1,t313_2,t313_3,t313_4,\
        t32,t33,t34,t35,t35_1,t35_2,t35_3,t35_4,t36,t37,t38,t38_1,t38_2,t39
    
    feature_3.drop_duplicates(inplace=True)
    
    ############# 4.user_merchant:  ##################3
    #用户领取商家的优惠券次数,user_id  merchant_id
    t41=dataset1[['user_id','merchant_id']]
    t41['4-1']=1
    t41=t41.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    
    #用户领取商家的优惠券后不核销次数,user_id  merchant_id
    t42=dataset1[(dataset1.date=='null')&(dataset1.coupon_id!='null')][['user_id','merchant_id']]
    t42['4-2']=1
    t42=t42.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    
    #用户领取商家的优惠券后核销次数,user_id  merchant_id
    t43=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id','merchant_id']]
    t43['4-3']=1
    t43=t43.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    
    #用户领取商家的优惠券后核销率,user_id  merchant_id
    t44=pd.merge(t41,t43,on=['user_id','merchant_id'],how='left')
    t44=t44.fillna(0)
    t44['4-4']=t44['4-3']/t44['4-1']
    t44=t44[['user_id','merchant_id','4-4']]
    
    #用户对每个商家的不核销次数占用户总的不核销次数的比重,user_id  merchant_id
    t45=dataset1[(dataset1.date=='null')&(dataset1.coupon_id!='null')][['user_id']]
    t45['4-5_1']=1
    t45=t45.groupby(['user_id']).agg('sum').reset_index()
    t45=pd.merge(t45,t42,on=['user_id'],how='left')
    t45['4-5']=t45['4-2']/t45['4-5_1']
    t45=t45[['user_id','merchant_id','4-5']]
    
    #用户对每个商家的优惠券核销次数占用户总的核销次数的比重,user_id  merchant_id
    t46=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id']]
    t46['4-6_1']=1
    t46=t46.groupby(['user_id']).agg('sum').reset_index()
    t46=pd.merge(t46,t43,on=['user_id'],how='left')
    t46['4-6']=t46['4-3']/t46['4-6_1']
    t46=t46[['user_id','merchant_id','4-6']]
    
    #用户对每个商家的不核销次数占商家总的不核销次数的比重,user_id  merchant_id
    t47=dataset1[(dataset1.date=='null')&(dataset1.coupon_id!='null')][['merchant_id']]
    t47['4-7_1']=1
    t47=t47.groupby(['merchant_id']).agg('sum').reset_index()
    t47=pd.merge(t47,t42,on=['merchant_id'],how='left')
    t47['4-7']=t47['4-2']/t47['4-7_1']
    t47=t47[['user_id','merchant_id','4-7']]
    
    #用户对每个商家的优惠券核销次数占商家总的核销次数的比重,user_id  merchant_id
    t48=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['merchant_id']]
    t48['4-8_1']=1
    t48=t48.groupby(['merchant_id']).agg('sum').reset_index()
    t48=pd.merge(t48,t43,on=['merchant_id'],how='left')
    t48['4-8']=t48['4-3']/t48['4-8_1']
    t48=t48[['user_id','merchant_id','4-8']]
    
    #特征四：user_merchant:
    feature_4=dataset1[['user_id','merchant_id']]
    feature_4.drop_duplicates(inplace=True)
    feature_4=pd.merge(feature_4,t41,on=['user_id','merchant_id'],how='left')
    feature_4=pd.merge(feature_4,t42,on=['user_id','merchant_id'],how='left')
    feature_4=pd.merge(feature_4,t43,on=['user_id','merchant_id'],how='left')
    feature_4=pd.merge(feature_4,t44,on=['user_id','merchant_id'],how='left')
    feature_4=pd.merge(feature_4,t45,on=['user_id','merchant_id'],how='left')
    feature_4=pd.merge(feature_4,t46,on=['user_id','merchant_id'],how='left')
    feature_4=pd.merge(feature_4,t47,on=['user_id','merchant_id'],how='left')
    feature_4=pd.merge(feature_4,t48,on=['user_id','merchant_id'],how='left')
    print (feature_4.shape)
    
    del t41,t42,t43,t44,t45,t46,t47,t48
    
    feature_4.drop_duplicates(inplace=True)
    
    ############# 5.coupon related:  ##################3
    #优惠券类型(直接优惠为0, 满减为1),coupon_id 
    t51=dataset1[['coupon_id','discount_rate']]
    t51['is_man_jian']=t51.discount_rate.apply(is_man_jian)
    t51=t51[['coupon_id','is_man_jian']]
    t51.drop_duplicates(inplace=True)
    
    #优惠券折率,coupon_id
    t52=dataset1[['coupon_id','discount_rate']]
    t52['calc_discount_rate']=t52.discount_rate.apply(calc_discount_rate)
    t52=t52[['coupon_id','calc_discount_rate']]
    t52.drop_duplicates(inplace=True)
    
    #满减优惠券的最低消费,coupon_id
    t53=dataset1[['coupon_id','discount_rate']]
    t53['get_discount_man']=t53.discount_rate.apply(get_discount_man)
    t53=t53[['coupon_id','get_discount_man']]
    t53.drop_duplicates(inplace=True)
    
    #历史出现次数,coupon_id
    t54=dataset1[['coupon_id']]
    t54['t5-4']=1
    t54=t54.groupby('coupon_id').agg('sum').reset_index()
    
    #历史核销次数,coupon_id
    t55=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['coupon_id']]
    t55['t5-5']=1
    t55=t55.groupby('coupon_id').agg('sum').reset_index()
    
    #历史核销率,coupon_id
    t56=pd.merge(t54,t55,on=['coupon_id'],how='left')
    t56=t56.fillna(0)
    t56['5-6']=t56['t5-5']/t56['t5-4']
    t56=t56[['coupon_id','5-6']]
    
    ##历史核销时间率
    ##历史核销时间的max/min/median/mean
    #t57
    
    #领取优惠券是一周的第几天,date_received
    t58=dataset1[(dataset1.coupon_id!='null')][['coupon_id','date_received']]
    t58['dt']=t58.date_received.apply(lambda x:datetime.strptime(str(x),'%Y%m%d'))
    t58['5-8']=t58['dt'].dt.weekday+1
    t58=t58[['date_received','5-8']]
    t58.drop_duplicates(inplace=True)
    
    #领取优惠券是一月的第几天,date_received
    t59=dataset1[(dataset1.coupon_id!='null')][['coupon_id','date_received']]
    t59['dt']=t59.date_received.apply(lambda x:datetime.strptime(str(x),'%Y%m%d'))
    t59['5-9']=t59['dt'].dt.day
    t59=t59[['date_received','5-9']]
    t59.drop_duplicates(inplace=True)
    
    #历史上用户领取该优惠券次数,user_id coupon_id
    t510=dataset1[(dataset1.coupon_id!='null')][['user_id','coupon_id']]
    t510['5-10']=1
    t510=t510.groupby(['user_id','coupon_id']).agg('sum').reset_index()
    t510.drop_duplicates(inplace=True)
    
    #历史上用户消费该优惠券次数,user_id coupon_id
    t511=dataset1[(dataset1.date!='null')&(dataset1.coupon_id!='null')][['user_id','coupon_id']]
    t511['5-11']=1
    t511=t511.groupby(['user_id','coupon_id']).agg('sum').reset_index()
    
    #历史上用户对该优惠券的核销率,user_id coupon_id 
    t512=pd.merge(t510,t511,on=['user_id','coupon_id'],how='left')
    t512=t512.fillna(0)
    t512['5-12']=t512['5-11']/t512['5-10']
    t512=t512[['user_id','coupon_id','5-12']]
    
    feature_5=dataset1[['user_id','coupon_id','date_received']]
    feature_5=pd.merge(feature_5,t51,on=['coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t52,on=['coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t53,on=['coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t54,on=['coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t55,on=['coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t56,on=['coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t58,on=['date_received'],how='left')
    feature_5=pd.merge(feature_5,t59,on=['date_received'],how='left')
    feature_5=pd.merge(feature_5,t510,on=['user_id','coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t511,on=['user_id','coupon_id'],how='left')
    feature_5=pd.merge(feature_5,t512,on=['user_id','coupon_id'],how='left')
    print(feature_5.shape)
    
    del t51,t510,t511,t512,t52,t53,t54,t55,t56,t58,t59
    
    feature_5.drop_duplicates(inplace=True)
    
    ############# 6.leakage:  ##################3
    #用户领取的所有优惠券数目,user_id
    t61=dataset1[['user_id']]
    t61['6-1']=1
    t61=t61.groupby('user_id').agg('sum').reset_index()
    
    #用户领取的特定优惠券数目,user_id coupon_id
    t62=dataset1[['user_id','coupon_id']]
    t62['6-2']=1
    t62=t62.groupby(['user_id','coupon_id']).agg('sum').reset_index()
    
    #用户此次之后/前领取的所有优惠券数目,user_id date_received    
    def retBef(s):
        dr,date_received=s.split(',')
        dates=date_received.split(':')
        res=[]
        for x in dates:
            if x<dr:
                res.append(x)
        return -len(res)
        
    t63=dataset1[['user_id','date_received']]
    t63.date_received=t63.date_received.astype('str')
    t63=t63.groupby(['user_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    t63_1=dataset1[['user_id','date_received']]
    t63_1.rename(columns={'date_received':'dr'},inplace=True)
    t63_1.dr=t63_1.dr.astype('str')
    t63=pd.merge(t63_1,t63,on=['user_id'],how='left')
    t63['6-3_1']=t63.dr.astype('str')+','+t63.date_received
    t63['6-3']=t63['6-3_1'].apply(retBef)
    t63=t63[['user_id','dr','6-3']]
    t63.rename(columns={'dr':'date_received'},inplace=True)
    t63.drop_duplicates(inplace=True)
    
    #用户此次之后/前领取的特定优惠券数目,user_id date_received
    def retAft(s):
        dr,date_received=s.split(',')
        dates=date_received.split(':')
        res=[]
        for x in dates:
            if x>dr:
                res.append(x)
        return len(res)
        
    t64=dataset1[['user_id','date_received']]
    t64.date_received=t64.date_received.astype('str')
    t64=t64.groupby(['user_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    t64_1=dataset1[['user_id','date_received']]
    t64_1.rename(columns={'date_received':'dr'},inplace=True)
    t64_1.dr=t64_1.dr.astype('str')
    t64=pd.merge(t64_1,t64,on=['user_id'],how='left')
    t64['6-4_1']=t64.dr.astype('str')+','+t64.date_received
    t64['6-4']=t64['6-4_1'].apply(retAft)
    t64=t64[['user_id','dr','6-4']]
    t64.rename(columns={'dr':'date_received'},inplace=True)
    t64.drop_duplicates(inplace=True)
    
    ##用户上/下一次领取的时间间隔
    #t65
    
    #用户领取特定商家的优惠券数目,user_id  merchant_id
    t66=dataset1[['user_id','merchant_id']]
    t66['6-6']=1
    t66=t66.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    
    #用户领取的不同商家数目,user_id  merchant_id 
    t67=dataset1[['user_id','merchant_id']]
    t67.drop_duplicates(inplace=True)
    t67['6-7']=1
    t67=t67.groupby(['user_id']).agg('sum').reset_index()
    
    #用户当天领取的优惠券数目,user_id date_received
    t68 = dataset1[['user_id','date_received']]
    t68['6-8'] = 1
    t68 = t68.groupby(['user_id','date_received']).agg('sum').reset_index()
    
    #用户当天领取的特定优惠券数目,user_id coupon_id date_received
    t69 = dataset1[['user_id','coupon_id','date_received']]
    t69['6-9'] = 1
    t69 = t69.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()
    
    #用户领取的所有优惠券种类数目,user_id
    t610=dataset1[['user_id','coupon_id']]
    t610.drop_duplicates(inplace=True)
    t610['6-10']=1
    t610=t610.groupby('user_id').agg('sum').reset_index()
    
    
    #商家被领取的优惠券数目,merchant_id
    t611=dataset1[['merchant_id']]
    t611['6-11']=1
    t611=t611.groupby('merchant_id').agg('sum').reset_index()
    
    #商家被领取的特定优惠券数目,merchant_id coupon_id
    t612=dataset1[['merchant_id','coupon_id']]
    t612['6-12']=1
    t612=t612.groupby(['merchant_id','coupon_id']).agg('sum').reset_index()
    
    #商家被多少不同用户领取的数目,merchant_id    user_id
    t613=dataset1[['merchant_id','user_id']]
    t613.drop_duplicates(inplace=True)
    t613['6-13']=1
    t613=t613.groupby('merchant_id').agg('sum').reset_index()
    
    #商家发行的所有优惠券种类数目,merchant_id
    t614=dataset1[['merchant_id','coupon_id']]
    t614.drop_duplicates(inplace=True)
    t614['6-14']=1
    t614=t614.groupby('merchant_id').agg('sum').reset_index()
    
    #不需要t615
    t615=dataset1[['user_id','coupon_id','date_received']]
    t615.date_received=t615.date_received.astype('str')
    t615=t615.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    t615['receive_number']=t615.date_received.apply(lambda s:len(s.split(':')))
    #t615=t615[t615.receive_number>1]
    t615['max_date_received']=t615.date_received.apply(lambda x:max([int(d) for d in x.split(':')]))
    t615['min_date_received']=t615.date_received.apply(lambda x:min([int(d) for d in x.split(':')]))
    t615=t615[['user_id','coupon_id','max_date_received','min_date_received']]
    #user_id coupon_id date_received
    t616=dataset1[['user_id','coupon_id','date_received']]
    t616=pd.merge(t616,t615,on=['user_id','coupon_id'],how='left')
    t616['616_1']=t616.max_date_received-t616.date_received.astype('int')
    t616['616_2']=t616.date_received.astype('int')-t616.min_date_received
    
    def is_firstlastone(x):
        if x==0:
            return 0
        else:
            return 1
    
    def returnfuyi(x):
        return -1
    
    t616['616_11']=t616['616_1'].apply(is_firstlastone)
    t616['616_21']=t616['616_2'].apply(is_firstlastone)
    
    t616['616_12']=t616[t616['max_date_received']==np.int64(t616['min_date_received'])]['616_11'].apply(returnfuyi)
    t616['616_22']=t616[t616['max_date_received']==np.int64(t616['min_date_received'])]['616_21'].apply(returnfuyi)
    t616=t616.fillna(0)
    t616['616_11']=t616['616_11']+t616['616_12']
    t616['616_21']=t616['616_21']+t616['616_22']
    t616=t616[['user_id', 'coupon_id', 'date_received', 'max_date_received',
           'min_date_received', '616_1', '616_2', '616_11', '616_21']]
    t616.drop_duplicates(inplace=True)
    
    t617=dataset1[['user_id','coupon_id','date_received']]
    t617.date_received=t617.date_received.astype('str')
    t617=t617.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
    t617.rename(columns={'date_received':'dates'},inplace=True)
    def get_day_gap_before(s):
        date_received,dates=s.split('-')
        dates=dates.split(':')
        gaps=[]
        for d in dates:
            this_gap=(date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
            if this_gap>0:
                gaps.append(this_gap)
        if len(gaps)==0:
            return -1
        else:
            return min(gaps)
            
    def get_day_gap_after(s):
        date_received,dates=s.split('-')
        dates=dates.split(':')
        gaps=[]
        for d in dates:
            this_gap=(date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
            if this_gap>0:
                gaps.append(this_gap)
        if len(gaps)==0:
            return -1
        else:
            return min(gaps)
    #user_id coupon_id date_received
    t618=dataset1[['user_id','coupon_id','date_received']]
    t618=pd.merge(t618,t617,on=['user_id','coupon_id'],how='left')
    t618['date_received_date']=t618.date_received.astype('str')+'-'+t618.dates
    t618['day_gap_before']=t618.date_received_date.apply(get_day_gap_before)
    t618['day_gap_after']=t618.date_received_date.apply(get_day_gap_after)
    t618=t618[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]
    t618.drop_duplicates(inplace=True)
    
    #t65,t615,t617不需要
    feature_6=dataset1[['user_id','coupon_id','merchant_id','date_received']]
    feature_6.drop_duplicates(inplace=True)
    feature_6=pd.merge(feature_6,t61,on=['user_id'],how='left')
    feature_6=pd.merge(feature_6,t62,on=['user_id','coupon_id'],how='left')
    feature_6=pd.merge(feature_6,t63,on=['user_id','date_received'],how='left')
    feature_6=pd.merge(feature_6,t64,on=['user_id','date_received'],how='left')
    feature_6=pd.merge(feature_6,t66,on=['user_id','merchant_id'],how='left')
    feature_6=pd.merge(feature_6,t67,on=['user_id','merchant_id'],how='left')
    feature_6=pd.merge(feature_6,t68,on=['user_id','date_received'],how='left')
    feature_6=pd.merge(feature_6,t69,on=['user_id','coupon_id','date_received'],how='left')
    feature_6=pd.merge(feature_6,t610,on=['user_id'],how='left')
    feature_6=pd.merge(feature_6,t611,on=['merchant_id'],how='left')
    feature_6=pd.merge(feature_6,t612,on=['merchant_id','coupon_id'],how='left')
    feature_6=pd.merge(feature_6,t613,on=['merchant_id','user_id'],how='left')
    feature_6=pd.merge(feature_6,t614,on=['merchant_id'],how='left')
    feature_6=pd.merge(feature_6,t616,on=['user_id','coupon_id','date_received'],how='left')
    feature_6=pd.merge(feature_6,t618,on=['user_id','coupon_id','date_received'],how='left')
    print(feature_6.shape)
    
    del t61,t610,t611,t612,t613,t614,t615,t616,t617,t618,t62,t63,t63_1,t64,t64_1,t66,t67,t68,t69
    
    feature_6.drop_duplicates(inplace=True)
    
    #得到总特征
    if label_dataset1.size==681840:
        feature=label_dataset1[['user_id','merchant_id','coupon_id','date_received']]
    else:
        feature=label_dataset1[['user_id','merchant_id','coupon_id','date_received','date']]
    feature.drop_duplicates(inplace=True)
    feature=pd.merge(feature,feature_1,on=['user_id','coupon_id','merchant_id'],how='left')
    feature=pd.merge(feature,feature_2,on=['user_id'],how='left')
    feature=pd.merge(feature,feature_3,on=['user_id','merchant_id','coupon_id','merchant_id'],how='left')
    feature=pd.merge(feature,feature_4,on=['user_id','merchant_id'],how='left')
    feature=pd.merge(feature,feature_5,on=['user_id','coupon_id','date_received'],how='left')
    feature=pd.merge(feature,feature_6,on=['user_id','coupon_id','merchant_id','date_received'],how='left')
    feature.drop_duplicates(inplace=True)
    print (feature.shape)
    
    return feature

    
#genFeature
def get_label(s):
    s = s.split(':')
    if s[0]=='null':
        return 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1
        
dat1=genFeature(feature1,on_dataset1,dataset1)    
print (dat1.shape)
dat1['day_of_week']=dat1['date_received'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d')).dt.weekday+1
dat1['is_weekend'] = dat1.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies=pd.get_dummies(dat1.day_of_week)
weekday_dummies.columns=['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dat1=pd.concat([dat1,weekday_dummies],axis=1)
dat1['label']=dat1.date.astype('str')+':'+dat1.date_received.astype('str')
dat1.label=dat1.label.apply(get_label)
dat1.to_csv('data/dat1.csv',index=None)

dat2=genFeature(feature2,on_dataset2,dataset2)
print (dat2.shape)
dat2['day_of_week']=dat2['date_received'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d')).dt.weekday+1
dat2['is_weekend'] = dat2.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies=pd.get_dummies(dat2.day_of_week)
weekday_dummies.columns=['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dat2=pd.concat([dat2,weekday_dummies],axis=1)
dat2['label']=dat2.date.astype('str')+':'+dat2.date_received.astype('str')
dat2.label=dat2.label.apply(get_label)
dat2.to_csv('data/dat2.csv',index=None)

dat3=genFeature(feature3,on_dataset2,dataset3)
print (dat3.shape)
dat3['day_of_week']=dat3['date_received'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d')).dt.weekday+1
dat3['is_weekend'] = dat3.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies=pd.get_dummies(dat3.day_of_week)
weekday_dummies.columns=['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dat3=pd.concat([dat3,weekday_dummies],axis=1)
dat3.to_csv('data/dat3.csv',index=None)

endtime = datetime.now()
run_time_cnt = (endtime - starttime).seconds
print('运行结束时间：{}，总运行时长：{}'.format(endtime.strftime('%Y-%m-%d %H:%M:%S'),run_time_cnt))



import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import time
#我的方法：[200]   train-auc:0.851069
#weapon的方法：[200]   train-auc:0.861169
#0407[200]   train-auc:0.861871
#merchant2_feature:[199]   train-auc:0.866483
dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset12 = pd.concat([dataset1,dataset2],axis=0)

dataset1_y = dataset1.label
dataset1_x = dataset1.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
dataset2_y = dataset2.label
dataset2_x = dataset2.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)

print (dataset1_x.shape,dataset2_x.shape,dataset3_x.shape)

#dataset1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
#dataset2 = xgb.DMatrix(dataset2_x,label=dataset2_y)
del dataset12_x['discount_rate']
del dataset3_x['discount_rate']
dataset12 = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3 = xgb.DMatrix(dataset3_x)

params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
     
print ("start cv:", time.strftime("%H:%M:%S",time.localtime()))
res = xgb.cv(params, dataset12, 200, nfold=10, early_stopping_rounds=50, verbose_eval=10)
print ("done cv:", time.strftime("%H:%M:%S",time.localtime()))
print ("best cv:", res['test-auc-mean'].tail(1).values[0])

#train on dataset1, evaluate on dataset2
#watchlist = [(dataset1,'train'),(dataset2,'val')]
#model = xgb.train(params,dataset1,num_boost_round=3000,evals=watchlist,early_stopping_rounds=300)

#watchlist = [(dataset12,'train')]
#model = xgb.train(params,dataset12,num_boost_round=20000,evals=watchlist)

watchlist = [(dataset12, 'train')]
model = xgb.train(params, dataset12, len(res), watchlist, verbose_eval=10)

#predict test set
dataset3_preds['label'] = model.predict(dataset3)
#dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label)
dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
print (dataset3_preds.describe())
    
#save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)

