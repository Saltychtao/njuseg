
.. contents::

1 Installation and Usage
------------------------

1.1 Installation
~~~~~~~~~~~~~~~~

Installation through pip:

.. code:: shell

    pip install njuseg

1.2 Usage
~~~~~~~~~

.. code:: python

    from njuseg import Segmenter
    segmenter = Segmenter.load_model(model_pth,use_gpu=True)
    sentences = ['美国联邦储备委员会 16 日发布的全国经济形势调查报告显示，去年 12 月初至今年 1 月上旬，美国经济继续温和扩张，但美国企业对经济前景的乐观程度有所下降。','美联储注意到了市场对全球经济放缓等风险因素的担心，但当前美国经济发生衰退的风险并未上升。']
    segmented_sentences = segmenter.seg(sentences)

2 Performance
-------------

2.1 In domain:
~~~~~~~~~~~~~~

with pretrained unigram + bigram embedding

.. table::

    +-------+-------+-------+-------+-------+
    |   PKU |   MSR |  CTB5 |  CTB6 | NLPCC |
    +=======+=======+=======+=======+=======+
    | 96.63 | 96.52 | 98.14 | 96.13 | 95.82 |
    +-------+-------+-------+-------+-------+

3 Speed
-------

On CPU: 20 k characters per second
On single NVIDIA GTX 1080 GPU: 160 k characters per second
