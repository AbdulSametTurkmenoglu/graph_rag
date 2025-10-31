# LlamaIndex ile Türkçe Haberlerden Bilgi Grafı (Graph RAG) Oluşturma

Bu proje, [LlamaIndex](https://www.llamaindex.ai/) kütüphanesini ve Anthropic (Claude) modelini kullanarak yaklaşık 42.000 Türkçe haber metninden oluşan bir veri setini işler. Projenin amacı, bu metinlerden bir Bilgi Grafı (Knowledge Graph) oluşturmak ve bu grafiği doğal dil sorgularıyla (RAG - Retrieval-Augmented Generation) sorgulamaktır.

Script, metinler arasındaki ilişkileri (üçlüleri - triplets) çıkarır, bir graf veritabanında (SimpleGraphStore) saklar ve bu grafiği sorgulamak için bir sorgu motoru (query engine) sağlar.

##  Temel Özellikler

* **Veri Yükleme:** `veri/42bin_haber/news` klasöründeki ~42.000 haber metnini okur ve bir Pandas DataFrame'e yükler.
* **Graph RAG:** LlamaIndex'in `KnowledgeGraphIndex` özelliğini kullanarak metinlerden otomatik olarak bilgi üçlüleri (özne, ilişki, nesne) çıkarır.
* **LLM Entegrasyonu:** İlişki çıkarımı için Anthropic `claude-sonnet-4-5-20250929` modelini kullanır.
* **Embedding Entegrasyonu:** Varlıkların (entities) anlamsal temsili için `BAAI/bge-small-en-v1.5` embedding modelini kullanır.
* **Doğal Dil Sorgulaması:** Oluşturulan bilgi grafı üzerinde doğal dil sorguları çalıştırır.
* **Görselleştirme:** `networkx` ve `matplotlib` kütüphanelerini kullanarak çıkarılan ilişkilerin bir kısmını görselleştirir ve bir `.png` dosyası olarak kaydeder.

##  Kullanılan Teknolojiler

* `llama-index` (Çekirdek, Anthropic LLM, SimpleGraphStore)
* `llama-index-embeddings-huggingface`
* `anthropic` (Claude API için)
* `pandas` (Veri yükleme için)
* `networkx` (Graf analizi ve görselleştirme)
* `matplotlib` (Graf çizimi)

##  Kurulum

1.  Gerekli Python kütüphanelerini yükleyin:

    ```bash
    pip install llama-index pandas networkx matplotlib llama-index-llms-anthropic llama-index-embeddings-huggingface torch
    ```

2.  Script'teki `ANTHROPIC_API_KEY` değişkenine geçerli bir Anthropic API anahtarı girmeniz gerekmektedir:

    ```python
    ANTHROPIC_API_KEY = "SIZIN_API_ANAHTARINIZ"
    ```

##  Nasıl Çalışır (İş Akışı)

1.  **Veri Yükleme:** `veri/42bin_haber/news` klasöründeki tüm `.txt` dosyaları taranır, okunur ve `kategori` ve `metin` sütunlarıyla bir Pandas DataFrame'e dönüştürülür.
2.  **Belge Hazırlama:** DataFrame'deki her bir satır, `llama_index.core.Document` nesnesine dönüştürülür.
3.  **Modellerin Başlatılması:** Anthropic LLM (`claude-sonnet-4-5-20250929`) ve HuggingFace embedding modeli (`BAAI/bge-small-en-v1.5`) yüklenir.
4.  **Bilgi Grafı Oluşturma:** `KnowledgeGraphIndex.from_documents` fonksiyonu çağrılır.
    * **Not:** Bu script'te tüm 42.000 belge yerine, örnek olarak sadece ilk 50 belge (`documents[:50]`) kullanılarak graf oluşturulmuştur.
    * LLM, bu 50 belgedeki metinleri analiz eder ve `max_triplets_per_chunk=1` parametresine göre ilişkileri (üçlüleri) çıkarır.
    * Bu üçlüler `SimpleGraphStore` içinde saklanır.
5.  **Sorgu Motoru:** Oluşturulan `kg_index` üzerinden bir sorgu motoru (`as_query_engine`) yaratılır.
6.  **Sorgulama:** Önceden tanımlanmış bir dizi (örn: "Dışişleri Bakanı Davutoğlu... hakkında ne söyledi?") sorgu motora gönderilir ve LLM'in graf bilgisini kullanarak ürettiği yanıtlar yazdırılır.
7.  **Görselleştirme:** Graf veritabanından (`graph_store`) tüm ilişkiler (edges) çıkarılır ve `networkx` kullanılarak bir yönlü grafiğe dönüştürülür. Bu graf `matplotlib` ile çizdirilir ve `bilgi_grafi.png` olarak kaydedilir.

##  Örnek Çıktılar

```text
Sorgu: Dışişleri Bakanı Davutoğlu Yunanistan ile Türkiye hakkında ne söyledi? Yanıt: Dışişleri Bakanı Davutoğlu, Yunanistan ile Türkiye arasındaki farklılıkların ortak vizyon ile çözülebileceğini söyledi. İki ülke arasında görüş ayrılıkları ve farklı yaklaşımlar olabileceğini belirten Davutoğlu, aynı vizyonun paylaşılmasıyla aradaki sorunların ve problemlerin daha rahat çözülebileceğini ifade etti. Ayrıca, Yunan mevkidaşı ile yaptığı görüşmelerden elde ettiği sonucun da bu yönde olduğunu vurguladı. Davutoğlu, birçok alanda ortak mutabakatların ilk tohumlarının atıldığını ve Ocak ayında her iki başbakanın eşbaşkanlığında Türkiye'de bir toplantı gerçekleşeceğini açıkladı.
```

```text
Sorgu: Borsa İstanbul hakkında neler yazıldı? Yanıt: Verilen bağlam bilgisinde Borsa İstanbul hakkında herhangi bir bilgi bulunmamaktadır. Sağlanan içerikte bu konuyla ilgili bir yazı veya bilgi yer almamaktadır.
```

```text
Sorgu: Faiz kararının etkileri nelerdir? Yanıt: Üzgünüm, verilen bağlam bilgisinde faiz kararının etkileri hakkında herhangi bir bilgi bulunmamaktadır. Bu soruyu yanıtlayabilmem için faiz kararları ve bunların ekonomik, finansal veya sosyal etkileri hakkında bilgi içeren kaynaklara ihtiyacım var.
```

```text
Sorunuzu yanıtlayabilmem için lütfen ilgili bağlam bilgisini sağlayın.
```

### Bilgi Grafiği Görseli

Çıkarılan ilişkileri gösteren graf görseli `bilgi_grafi.png` dosyası olarak kaydedilir.

![Haber Metinlerinden Çıkarılan İlişkiler Grafı](bilgi_grafi.png)

### Sorgu Yanıtları

Script, bilgi grafiğini kullanarak aşağıdaki gibi yanıtlar üretir:
