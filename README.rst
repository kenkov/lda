==============================
LDA
==============================

Collapsed Gibbs Sampling を使った LDA の実装です。

例
===

単語分割されたテキストファイルを渡すと各々の
文のトピックベクトルを返します。

例えば附属の ``test.txt``

::

    a b c
    a b d
    c d e f
    a b d e
    a b c d e

を ``lda.py`` にわたすと、モデルファイル ``test.model`` を生成して
以下のような結果になります。

.. code-block:: bash

    $ python lda.py test.txt -n2
     2015-02-07 18:18:58,654 - lda.py - INFO - train model: num_d=5, num_w=6, num_z=2
     2015-02-07 18:18:58,691 - lda.py - INFO - save model in test.model: num_d=5, num_w=6, num_z=2
     2015-02-07 18:18:58,691 - lda.py - INFO - load model from test.model: num_d=5, num_w=6, num_z=2
     [[ 0.42857143  0.57142857]
      [ 0.28571429  0.71428571]
      [ 0.5         0.5       ]
      [ 0.25        0.75      ]
      [ 0.44444444  0.55555556]]

``-n`` オプションでトピック数を指定します。デフォルトのトピック数は ``2`` です。
