# hierarchical-detox
This repository contains code for Hierarchical Adversarial Correction to Mitigate Identity Term Bias in Toxicity Detection

**This repository is still under construction**

# Citation # 

If you use any of the material from this repository, please cite our paper: [Johannes Schäfer, Ulrich Heid, and Roman Klinger. 2024. Hierarchical Adversarial Correction to Mitigate Identity Term Bias in Toxicity Detection. In Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis, pages 35–51, Bangkok, Thailand. Association for Computational Linguistics.](https://aclanthology.org/2024.wassa-1.4/)

BibTeX:
```text
@inproceedings{schafer-etal-2024-hierarchical,
    title = "Hierarchical Adversarial Correction to Mitigate Identity Term Bias in Toxicity Detection",
    author = {Sch{\"a}fer, Johannes  and
      Heid, Ulrich  and
      Klinger, Roman},
    editor = "De Clercq, Orph{\'e}e  and
      Barriere, Valentin  and
      Barnes, Jeremy  and
      Klinger, Roman  and
      Sedoc, Jo{\~a}o  and
      Tafreshi, Shabnam",
    booktitle = "Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, {\&} Social Media Analysis",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wassa-1.4",
    pages = "35--51",
    abstract = "Corpora that are the fundament for toxicity detection contain such expressions typically directed against a target individual or group, e.g., people of a specific gender or ethnicity. Prior work has shown that the target identity mention can constitute a confounding variable. As an example, a model might learn that Christians are always mentioned in the context of hate speech. This misguided focus can lead to a limited generalization to newly emerging targets that are not found in the training data. In this paper, we hypothesize and subsequently show that this issue can be mitigated by considering targets on different levels of specificity. We distinguish levels of (1) the existence of a target, (2) a class (e.g., that the target is a religious group), or (3) a specific target group (e.g., Christians or Muslims). We define a target label hierarchy based on these three levels and then exploit this hierarchy in an adversarial correction for the lowest level (i.e. (3)) while maintaining some basic target features. This approach does not lower the toxicity detection performance but increases the generalization to targets not being available at training time.",
}
```
