# DeepLearningTest
Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks? (FSE 2020)

## Summary 
Recent effort to test deep learning systems has produced an intuitive and compelling test criterion called neuron coverage (NC), which resembles the notion of traditional code coverage. NC measures the proportion of neurons activated in a neural network and it is implicitly assumed that increasing NC improves the quality of a test suite. In an attempt to automatically generate a test suite that increases NC, we design a novel diversity promoting regularizer that can be plugged into existing adversarial attack algorithms. We then assess whether such attempts to increase NC could generate a test suite that (1) detects adversarial attacks successfully, (2) produces natural inputs, and (3) is unbiased to particular class predictions. Contrary to expectation, our extensive evaluation finds that increasing NC actually makes it harder to generate an effective test suite: higher
neuron coverage leads to fewer defects detected, less natural inputs, and more biased prediction preferences. Our results invoke skepticism that increasing neuron coverage may not be a meaningful objective for generating tests for deep neural networks and call for a new test generation technique that considers defect detection, naturalness, and output impartiality in tandem.

## Team 
This project is developed by Professor [Miryung Kim](http://web.cs.ucla.edu/~miryung/)'s Software Engineering and Analysis Laboratory at UCLA. 
If you encounter any problems, please open an issue or feel free to contact us:

[Fabrice Harel-Canada](https://fabrice.harel-canada.com/): PhD student, fabricehc@cs.ucla.edu;

[Lingxiao Wang](https://scholar.google.com/citations?user=VPyxd6kAAAAJ&hl=zh-CN): PhD student, lingxw@cs.ucla.edu;

[Muhammad Ali Gulzar](https://people.cs.vt.edu/~gulzar/): Assistant Professor, gulzar@cs.vt.edu;

[Quanquan Gu](http://web.cs.ucla.edu/~qgu/): Assistant Professor, qgu@cs.ucla.edu;

[Miryung Kim](https://web.cs.ucla.edu/~miryung/): Professor, miryung@cs.ucla.edu;

## How to cite 
Please refer to our FSE'20 paper, [Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks?](https://web.cs.ucla.edu/~miryung/Publications/fse2020-testingdeeplearning.pdf)

### Bibtex  
```
@article{HarelCanada2020IsNC,
  title={Is neuron coverage a meaningful measure for testing deep neural networks?},
  author={F. Harel-Canada and L. Wang and Muhammad Ali Gulzar and Quanquan Gu and Miryung Kim},
  journal={Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  year={2020}
}
```

[DOI Link](https://dl.acm.org/doi/10.1145/3368089.3409754)
