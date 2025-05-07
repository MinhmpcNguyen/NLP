# from gensim.models import KeyedVectors

# # Sau khi tải về và giải nén:
# model = KeyedVectors.load_word2vec_format("cc.vi.300.vec", binary=False)
# model.save("cc.vi.300.kv")
import fasttext.util

fasttext.util.download_model("vi", if_exists="ignore")  # English
ft = fasttext.load_model("cc.vi.300.bin")
