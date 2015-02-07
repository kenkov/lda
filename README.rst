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

    2015-02-07 18:05:13,694 - lda.py - INFO - num_d: 5, num_w: 6, num_z:2
    2015-02-07 18:05:13,728 - lda.py - INFO - save model in test.model: num_d=5, num_w=6, num_z=2
    2015-02-07 18:05:13,728 - lda.py - INFO - load model from : num_d=test.model, num_w=5, num_z=6
    [[ 0.57142857  0.42857143]
     [ 0.42857143  0.57142857]
     [ 0.625       0.375     ]
     [ 0.25        0.75      ]
     [ 0.66666667  0.33333333]]
