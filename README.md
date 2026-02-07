# Project Summary

If you're new here, this project was started in response to [an effort to extract some unredacted content in the Epstein archives](https://neosmart.net/blog/recreating-epstein-pdfs-from-raw-encoded-attachments/). The CNN in this repo was used to successfully exfiltrate [DBC12.pdf](https://archive.org/details/dbc-12-one-page-invite-with-reply) from [EFTA00400459](https://archive.org/details/efta-00400459_pages).

You can read about this code/approach here: [EFTA00400459 has been cracked, DBC12.pdf liberated](https://neosmart.net/blog/efta00400459-has-been-cracked-dbc12-pdf-liberated/).

The code in this project runs against the images extracted from the PDF with `pdfimages`, you can download [an archive containing them here](https://archive.org/details/efta-00400459_pages).

### Basic Usage Info

Expects `../EFTA00400459-{000..=075}_2x.png` to exist

* Run `./train.sh` to generate training from train_top.txt and train_bot.txt corresponding to page-001_2x.png
* Run `./run.sh` to OCR all pages and generated recovered.pdf

Trains from top of page-001 and bottom of page-001 non-contiguously to capture vertical drift.
Memorizes grid location and reuses for subsequent pages (non-training runs) to prevent pixel shifts.

In training runs with `-d`/`--debug`, generates a debug view that lets you see if you mis-typed anything by showing greatest outliers compared to the rest of the members assigned to the bucket:

![Typo sanity checking when training](./img/training-no-typos.png)

In inference runs, generates a debug view (when `-d` is in use with no `-q`/`--quiet`) that shows the max outliers compared to the rest of the characters in the image. When `-o`/`--output` is specified, the debug view is saved to `<basename>-proof.png` so you can inspect it later.

![Post-inference analysis](./img/inference-analysis.png)
