# Makalede yapmak istediğimizi tam ifade edememişiz
# global plan ve local plan arasındaki farkı net bir şekilde ifade etmek lazım.
# sürekli aksiyon uzayı veren DDPG, A3C, PPO algoritmaları ile karşılaştır. Niye bunu kullanamdık. Future work e ekleyelim.
# Durum Uzayı ve aksiyon uzayını neye göre karar verdik. Matematiksel olarak ifade edebilir miyiz? Pure pursuit yaklaşımından getirdik.
# away path-> on path gibi MDP modeli olarak oluştur. 
# MDP future a bakıyor. geçmiş veri heuristic olarak eklenebilir mi? (A* , DFS)
# # curriculum learning karşılaştırması yap (Öncelik verme, öncelik ver). Karşılaştır. 
# Figure 5 i detaylandır.
# Modellerin karşılaştırmasını tablo olrak ekle.
# TEB ve DWA nın state of art algoritma olduğunu vurgula.


# Eksiklikleri nasıl düzelttik. Onları anlatalım.
# niye -> 0.8 veya 0.4 [0.4*(exp(observations[0]+0.8)-2.22554092849) ] sebebini açıkla. e^x e ve linear ile karşılaştır.
# zyaptığım değişiklikleri anlatalım.abs

## METRIC: robotun yola olan mesafesinin alanı, topam ödül/adım sayısı



# Literatür özetini tarihsel sürecini özetle. RL tarfında. Navigasyon tarfında. Son olrak rl + navigasyon
# (IEEE + springer + ACM) journal. Survey -> journal -> konferans(IROS ICRA) 
#
# Mevcuttaki uygulamaların anlatımı. TEB ve DWA detaya girelim. Orjinal makaleye mutlaka referans verelim.
# RL taradında: Nedir, nerede kullanılır, RL deki yeri(supervised, unsupervised ayrışan tarafı,) RL deki yaklaşımlar, RL deki tüm konuları kapsamlı anlatalım. 
#
# Önerilen Yöntem Başlığı
# Algoritmik akış olsun. Eğitim ve test aşamasının algoritmik akışı olsun.
# experimental setup
# Deneysel Sonuçlar
# İşin detayına giriyoruz. kaç episode eğittik. Derin öğrenme eğitimini nasıl yaptık  vs. Ve deneyesel sonuçlar.
# İmgeleri aynı boyutta cropla
# Deneysel sonuçlar/ Discussion(Tablolar, gerçeklenen trajectory koyduk onu yorumla.  Sonuçları yorumla). İlk olrak bire bir karşılaştırma en son overall değerlendirme.
# TEB ve DWA için gerçek robot sonuçlarını ekle.
# Sonuçlar. Çalışmamızı, deneysel sonuçları, discussionı özetle. Future work e dair  bir şeyler yaz.
# Modeldeki durum uzayı daraltılacak. ön-10m yan ve arka 2m.


###############################################
# 1. adım
# L,Theta, uzaklık için CNN ile eğitmek. aksiyon: 4-5
# Scan = max

# 2.adım
# 3m düşürülecek. 4-5 . 20 aksiyon. aksiyon:6 ,9 ->  max: 0.5 w: 0.4

# 3. adım
# ddgp ve td3 algoritmalarını test et.

# 4. adım
# continous action uzayı ile test edilecek. 2 output olacak. 1. linear 2. angular tahmini


# loss niye artmış. onun açıklamasını yaazalım