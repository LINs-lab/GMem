# GMem: Generative Modeling with Explicit Memory

![Teaser image](./docs/selected_pics.png)

[Preprint] Generative Modeling with Explicit Memory <br>
Yi Tang, Peng Sun, Zhenglin Cheng, Tao Lin


### Abstract
  Recent studies indicate that the denoising process in deep generative diffusion models implicitly learns and memorizes semantic information from the data distribution. These findings suggest that capturing more complex data distributions requires larger neural networks, leading to a substantial increase in computational demands, which in turn become the primary bottleneck in both training and inference of diffusion models.
  To this end, we introduce **G**enerative **M**odeling with **E**xplicit **M**emory **GMem**, leveraging an external memory bank in both training and sampling phases of diffusion models. This approach preserves semantic information from data distributions, reducing reliance on neural network capacity for learning and generalizing across diverse datasets. The results are significant: our **GMem** enhances both training, sampling efficiency, and generation quality. For instance, on ImageNet at $256 \times 256$ resolution, **GMem** accelerates SiT training by over $46.7\times$, achieving the performance of a SiT model trained for $7 M$ steps in fewer than $150K$ steps. Compared to the most efficient existing method, REPA, **GMem** still offers a $16\times$ speedup, attaining an FID score of 5.75 within $250K$ steps, whereas REPA requires over $4M$ steps. Additionally, our method achieves state-of-the-art generation quality, with an FID score of **3.56** without classifier-free guidance on ImageNet $256\times256$.

---


### System Requirements

- **Python and PyTorch:**
  - 64-bit Python 3.10 or later.
  - PyTorch 2.4.0 or later (earlier versions might work but are not guaranteed).

- **Additional Python Libraries:**
  - A complete list of required libraries is provided in the [requirements.txt](./requirements.txt) file.
  - To install them, execute the following command:
    ```bash
    pip install -r requirements.txt
    ```

---

### Getting Started

To reproduce the primary results from the paper, run the following script:

```bash
bash scripts/sample-gmem-xl.sh
```

This is a minimal standalone script that loads the best pre-trained model and generates 50K images.

---

### Pre-trained Models and Memory Bank

We offer the following pre-trained models and memory bank here:

#### Model Checkpoints
| Model Backbone       | Training Steps | File Location                |
|----------------------|----------------|------------------------------|
| SiT-XL/2             | 2M             | [Download Here](#)           |

#### Memory Bank
| Dataset              | Resolution     | Snippets         | Training Epo.  | File Location                |
|----------------------|----------------|------------------|----------------|------------------------------|
| ImageNet             | 256×256        | 640,000          | 5              | [Download Here](#)           |

**Important:** Ensure that both `bank.pth` and `bank.freq` are saved in the same directory to enable proper functionality.

---

### Additional Information

- Up next: the training code and scripts for GMem.

---

### Acknowledgement

This code is mainly built upon [SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2), and [REPA](https://github.com/sihyun-yu/REPA) repositories.


### BibTeX

```bibtex
```

