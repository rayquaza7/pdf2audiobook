import locale
import os
import re
import subprocess
import tempfile

import modal
from fastapi import UploadFile
from modal import method, web_endpoint

locale.getpreferredencoding = lambda: "UTF-8"


def preprocess_char(text, lang=None):
    """
    Special treatement of characters in certain languages
    """
    if lang == "ron":
        text = text.replace("ț", "ţ")
    return text


class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [
            x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()
        ]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        """
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = "perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        return txt_filt


def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None):
    txt = preprocess_char(txt, lang=lang)
    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"
    if is_uroman:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            txt = text_mapper.uromanize(txt, uroman_pl)
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt


image = modal.Image.debian_slim(
    python_version="3.10",
).from_dockerfile("Dockerfile")

stub = modal.Stub(
    "pdf2audiobook",
    image=image,
)


if stub.is_inside():
    import torch
    from commons import intersperse
    from models import SynthesizerTrn
    from utils import get_hparams_from_file, load_checkpoint


@stub.cls(gpu="any", timeout=1200)
class MMSTTS:
    def __enter__(self):
        ckpt_dir = "./eng"

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        vocab_file = f"{ckpt_dir}/vocab.txt"
        config_file = f"{ckpt_dir}/config.json"
        hps = get_hparams_from_file(config_file)
        text_mapper = TextMapper(vocab_file)
        net_g = SynthesizerTrn(
            len(text_mapper.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        )
        net_g.to(device)
        _ = net_g.eval()

        g_pth = f"{ckpt_dir}/G_100000.pth"

        _ = load_checkpoint(g_pth, net_g, None)
        self.net_g = net_g
        self.text_mapper = text_mapper
        self.hps = hps
        self.device = device

    @method()
    def generate(self, txt: str):
        txt = preprocess_text(txt, self.text_mapper, self.hps, lang="eng")
        stn_tst = self.text_mapper.get_text(txt, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            hyp = (
                self.net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                )[0][0, 0]
                .cpu()
                .float()
                .numpy()
            )

        print("Generated audio")
        return hyp, self.hps.data.sampling_rate


@stub.function()
@web_endpoint(
    method="POST",
)
def main(file: UploadFile):
    import fitz
    import numpy as np

    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    texts = []
    for page in doc:
        txt = page.get_text("text")
        texts.append(txt)

    audio = np.array([])
    sample_rate = 22050

    try:
        for result in MMSTTS().generate.map(texts):
            audio = np.concatenate([audio, result[0]])
            sample_rate = result[1]
    except Exception as e:
        print(e)

    return {
        "audio": audio.tolist(),
        "sample_rate": sample_rate,
    }
