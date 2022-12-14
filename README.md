# 🎓 Final dissertation for MSc in Artificial Intelligence 📄

This repository contains a final thesis realized for the [Master's degree in Artificial Intelligence](https://corsi.unibo.it/2cycle/artificial-intelligence), University of Bologna.

## Abstract

With the advent of high-performance computing devices, deep neural networks have gained a lot of popularity in solving many Natural Language Processing tasks. However, they are also vulnerable to adversarial attacks, which are able to modify the input text in order to mislead the target model. Adversarial attacks are a serious threat to the security of deep neural networks, and they can be used to craft adversarial examples that steer the model towards a wrong decision. In this dissertation, we propose SynBA, a novel contextualized synonym-based adversarial attack for text classification. SynBA is based on the idea of replacing words in the input text with their synonyms, which are selected according to the context of the sentence. We show that SynBA successfully generates adversarial examples that are able to fool the target model with a high success rate. We demonstrate three advantages of this proposed approach: (1) effective - it outperforms state-of-the-art attacks by semantic similarity and perturbation rate, (2) utility-preserving - it preserves semantic content, grammaticality, and correct types classified by humans, and (3) efficient - it performs attacks faster than other methods.

## Citing SynBA

If you use SynBA for your research, please cite as follows:
```
@phdthesis{amslaurea27348,
           title = {SynBA: A contextualized Synonym-Based adversarial Attack for text classification},
          author = {Giuseppe Murro},
        keywords = {Adversarial Machine Learning,NLP,Adversarial examples,Text classification,SynBA},
             url = {http://amslaurea.unibo.it/27348/}
}
```

The full text of the thesis is available at this [link](https://amslaurea.unibo.it/27348/) for online dissemination and communication through AMSLaurea, the institutional archive of the University of Bologna. Otherwise it is also accessible at this [path](tesi.pdf).


## Slideshow

Slides presented during the thesis discussion held on campus on 6th December 2022 can be retrieved from this [directory](./presentation/).

## LaTeX compilation
If you want to re-compile the LaTeX code or use it as template for your own thesis, you need to have a LaTeX distribution installed on your machine. I recommend [TeX Live](https://www.tug.org/texlive/).
Follow this [tutorial](https://mathjiajia.github.io/vscode-and-latex/#step-1-download--install-tex-live) to configure LaTeX with VSCode.

## Author

| Reg No. |   Name   | Surname |               Email               |                       Username                        |
| :-----: | :------: | :-----: | :-------------------------------: | :---------------------------------------------------: |
| 997317  | Giuseppe |  Murro  | `giuseppe.murro@studio.unibo.it`  |         [_gmurro_](https://github.com/gmurro)         |

