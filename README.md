# GMem: Generative Modeling with Explicit Memory

![Teaser image](./docs/selected_pics.png)

<div align="center">
  Yi Tang :man_student:, <a href="https://sp12138.github.io/">Peng Sun :man_artist:</a>, Zhenglin Cheng :man_student:, <a href="https://tlin-taolin.github.io/">Tao Lin :skier:</a>

  <a href="https://arxiv.org/abs/2412.08781">[arXiv] :page_facing_up:</a> | <a href="#bibliography">[BibTeX] :label:</a>
</div>


### Abstract
  Recent studies indicate that the denoising process in deep generative diffusion models implicitly learns and memorizes semantic information from the data distribution. These findings suggest that capturing more complex data distributions requires larger neural networks, leading to a substantial increase in computational demands, which in turn become the primary bottleneck in both training and inference of diffusion models.
  To this end, we introduce **G**enerative **M**odeling with **E**xplicit **M**emory **GMem**, leveraging an external memory bank in both training and sampling phases of diffusion models. This approach preserves semantic information from data distributions, reducing reliance on neural network capacity for learning and generalizing across diverse datasets. The results are significant: our **GMem** enhances both training, sampling efficiency, and generation quality. For instance, on ImageNet at $256 \times 256$ resolution, **GMem** accelerates SiT training by over $46.7\times$, achieving the performance of a SiT model trained for $7 M$ steps in fewer than $150K$ steps. Compared to the most efficient existing method, REPA, **GMem** still offers a $16\times$ speedup, attaining an FID score of 5.75 within $250K$ steps, whereas REPA requires over $4M$ steps. Additionally, our method achieves state-of-the-art generation quality, with an FID score of **3.56** without classifier-free guidance on ImageNet $256\times256$.

---


### Requirements

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

To reproduce the results from the paper, run the following script:

```bash
bash scripts/sample-gmem-xl.sh
```

**Important:** make sure to change `--ckpt` to correct path.

---

### Pre-trained Models and Memory Bank

We offer the following pre-trained model and memory bank here:

#### GMem Checkpoints
|    Backbone    | Training Steps | Dataset                   | Bank Size | Training Epo. | Download |
|----------------|----------------|---------------------------|-----------|---------------|----------|
| SiT-XL/2       | 2M             | ImageNet $256\times 256$  | 640,000   | 5             | [Huggingface](https://huggingface.co/Tangentone/GMem) |

---

### Additional Information

- Up next: the training code and scripts for GMem.

---

### Bibliography

If you find this repository helpful for your project, please consider citing our work:

```bibtex
@article{tang2024generative,
  title={Generative Modeling with Explicit Memory},
  author={Tang, Yi and Sun, Peng and Cheng, Zhenglin and Lin, Tao},
  journal={arXiv preprint arXiv:2412.08781},
  year={2024}
}
```


### Acknowledgement

This code is mainly built upon [SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2), and [REPA](https://github.com/sihyun-yu/REPA) repositories.


