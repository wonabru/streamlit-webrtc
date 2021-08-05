from streamlit_webrtc.frodoKEM.frodokem import FrodoKEM


class CPqc(FrodoKEM):

    def __init__(self, variant):
        super(CPqc, self).__init__(variant)
        print('Generating post Quantum cryptographic Key Pair. Wait a moment...')
        PUBLIC_KEY, PRIVATE_KEY = self.kem_keygen()
        self.PUBLIC_KEY = PUBLIC_KEY
        self.PRIVATE_KEY = PRIVATE_KEY
        print("pk =", PUBLIC_KEY.hex().upper())

        self.SHARED_SECRET = None

    def set_shared_secret(self, secret):
        self.SHARED_SECRET = secret

    def encap_secret(self, public_key):

        ciphertext, secret = self.kem_encaps(public_key)
        self.SHARED_SECRET = secret
        return ciphertext

    def decap_secret(self, ciphertext):
        return self.kem_decaps(self.PRIVATE_KEY, ciphertext)
