

#####CUSTOMER SEGMENTATION USING RFM#####

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.
#
# Şirket, ortak davranışlar sergileyen müşteri segmentleri
# özelinde pazarlama çalışmaları yapmanın gelir artışı
# sağlayacağını düşünmektedir.
#
# Örneğin şirket için çok kazançlı olan müşterileri elde tutmak
# için farklı kampanyalar, yeni müşteriler için farklı
# kampanyalar düzenlenmek istenmektedir.
#
# Veri Seti Hikayesi
#
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
#
# Online Retail II isimli veri seti İngiltere merkezli online bir satış
# mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını
# içermektedir.
#
# Bu şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır
#
# Şirketin müşterilerinin büyük çoğunluğu kurumsal müşterilerdir.
#
# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


########
#GÖREV-1: VERİYİ ANLAMA VE HAZIRLAMA
########

#1:Online Retail II excelindeki 2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’in kopyasını oluşturunuz.

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.head()

#2:Veri setinin betimsel istatistiklerini inceleyiniz.

df.shape
df.describe().T

#3:Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?

df.isnull().sum()

#4:Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.

df.dropna(inplace=True)
df.isnull().sum()

#5:Eşsiz ürün sayısı kaçtır?

df["Description"].nunique()

#6:Hangi üründen kaçar tane vardır?

df["Description"].value_counts().head(5)

#7:En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız.

df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending=False).head(5)

#8:Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.

df[~df["Invoice"].str.contains("C", na=False)].head()

#9:Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz.

df["TotalPrice"] = df["Quantity"] * df["Price"]

########
#GÖREV-2: RFM metriklerinin hesaplanması
########

#Recency:En son satın alma yaptığı zamandan bugüne geçen zaman.
#Frequency:Sıklık.Toplam satın alma sayısı
#Monetary:Toplam bırakılan kazanç.

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby("Customer ID").agg({"InvoiceDate":lambda InvoiceDate: (today_date-InvoiceDate.max()).days,
                                  "Invoice":lambda Invoice: Invoice.nunique(),
                                  "TotalPrice": lambda TotalPrice:TotalPrice.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']
rfm=rfm[(rfm["monetary"]>0)]
rfm.head()

########
#GÖREV-3:RFM skorlarının oluşturulması ve tek bir değişkene çevrilmesi
########

#Recency
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

#Frequency
rfm["frequency_score"]=pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])

#Monetary
rfm["monetary_score"]=pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])

rfm["RFM_SCORE"]=(rfm["recency_score"].astype(str)+
                  rfm["frequency_score"].astype(str))
rfm.head()

rfm[rfm["RFM_SCORE"] == "55"].head()

########
#GÖREV-4:RFM skorlarının segment olarak tanımlanması
########

seg_map={
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}

rfm["segment"]=rfm["RFM_SCORE"].replace(seg_map,regex=True)
rfm

########
#GÖREV-5:
########

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# new_customers: Yeni müşteriler. Yeni müşterilere kendimizi olabildiğince iyi tanıtmalıyız, burdan alışveriş yapmayı
# cazip hale getirmeliyiz. İlk alışverişlere özel indirim yaparak frequency arttırarak müşterileri bağlamalıyız.

# at_risk: Risk altındaki müşteriler. Bu müşterilerin recency değerleri yüksek yani son alışveriş yapma gününden bu güne
# çok zaman geçmiştir. Bunlara alışveriş yapmayı hatırlamalı ve bu müşterileri tekrar kazanmaya çalışılmalı.

# champions: Bu müşteriler son günlerde en sık satım alma gerçekleştiren ve aynı zamanda en çok parayı harcayan müşterilerimizdir.
# Bu segmentten düşmemeleri için daha çok önemsenmeli ve özel indirimlerle şirkete bağlı tutmaya çalışılmalı.


rfm[rfm["segment"] == "loyal_customers"].index
new_df = pd.DataFrame()
new_df["loyal_customer_id"] = rfm[rfm["segment"] == "loyal_customers"].index
new_df.head()
new_df.to_excel("loyal_customers.xls")