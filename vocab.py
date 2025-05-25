from nemo.collections.asr.models import EncDecCTCModelBPE

# Load the trained model
asr_model = EncDecCTCModelBPE.restore_from("/Users/saani/Mvaak AI/stt_hi_conformer_ctc_medium.nemo")

# Extract vocabulary
vocab = asr_model.decoder.vocabulary

# Inspect the first few tokens
print("Vocabulary sample:", vocab[:10])
print("Total vocab size:", len(vocab))
print(vocab)