{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.preprocessing.sequence import pad_sequences"
      ],
      "metadata": {
        "id": "3YibewsHw8eD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 540
        },
        "id": "IcamEtZnsqdu",
        "outputId": "1b59703a-cc67-46fc-ec5d-e9c8d752078f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "FileNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-5-b43e29bce5ab>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'spam.csv'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mdelimiter\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m','\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mencoding\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'latin-1'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mdf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/util/_decorators.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    309\u001b[0m                     \u001b[0mstacklevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mstacklevel\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    310\u001b[0m                 )\n\u001b[0;32m--> 311\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    312\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    313\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36mread_csv\u001b[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)\u001b[0m\n\u001b[1;32m    584\u001b[0m     \u001b[0mkwds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkwds_defaults\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    585\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 586\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    587\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    588\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m_read\u001b[0;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[1;32m    480\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    481\u001b[0m     \u001b[0;31m# Create the parser.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 482\u001b[0;31m     \u001b[0mparser\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    483\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    484\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[1;32m    809\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"has_index_names\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"has_index_names\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    810\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 811\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    812\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    813\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[0;34m(self, engine)\u001b[0m\n\u001b[1;32m   1038\u001b[0m             )\n\u001b[1;32m   1039\u001b[0m         \u001b[0;31m# error: Too many arguments for \"ParserBase\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1040\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mmapping\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# type: ignore[call-arg]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1041\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1042\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_failover_to_python\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/io/parsers/c_parser_wrapper.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, src, **kwds)\u001b[0m\n\u001b[1;32m     49\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     50\u001b[0m         \u001b[0;31m# open handles\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 51\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_open_handles\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msrc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     52\u001b[0m         \u001b[0;32massert\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandles\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     53\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/io/parsers/base_parser.py\u001b[0m in \u001b[0;36m_open_handles\u001b[0;34m(self, src, kwds)\u001b[0m\n\u001b[1;32m    227\u001b[0m             \u001b[0mmemory_map\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"memory_map\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    228\u001b[0m             \u001b[0mstorage_options\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"storage_options\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 229\u001b[0;31m             \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"encoding_errors\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"strict\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    230\u001b[0m         )\n\u001b[1;32m    231\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/pandas/io/common.py\u001b[0m in \u001b[0;36mget_handle\u001b[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[1;32m    705\u001b[0m                 \u001b[0mencoding\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mioargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mencoding\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    706\u001b[0m                 \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merrors\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 707\u001b[0;31m                 \u001b[0mnewline\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"\"\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    708\u001b[0m             )\n\u001b[1;32m    709\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'spam.csv'"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)\n",
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kLabjSHVs4uc",
        "outputId": "08a40afd-043e-4732-e5b0-1cafb4151d2a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 5572 entries, 0 to 5571\n",
            "Data columns (total 2 columns):\n",
            " #   Column  Non-Null Count  Dtype \n",
            "---  ------  --------------  ----- \n",
            " 0   v1      5572 non-null   object\n",
            " 1   v2      5572 non-null   object\n",
            "dtypes: object(2)\n",
            "memory usage: 87.2+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(df.v1)\n",
        "plt.xlabel('Label')\n",
        "plt.title('Number of ham and spam messages')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 367
        },
        "id": "qrZPLawvtKul",
        "outputId": "b2b732dd-128c-433e-8be2-054eafa41889"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  FutureWarning\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Number of ham and spam messages')"
            ]
          },
          "metadata": {},
          "execution_count": 17
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PXdUorCrsPM3"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from keras.models import Model\n",
        "from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding\n",
        "from keras.optimizers import RMSprop\n",
        "from keras.preprocessing.text import Tokenizer\n",
        "from keras.preprocessing import sequence\n",
        "from keras.utils import to_categorical\n",
        "from keras.callbacks import EarlyStopping\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.v2\n",
        "Y = df.v1\n",
        "le = LabelEncoder()\n",
        "Y = le.fit_transform(Y)\n",
        "Y = Y.reshape(-1,1)"
      ],
      "metadata": {
        "id": "fU9V3JYytMXh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)"
      ],
      "metadata": {
        "id": "_tL5j8upt_vd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "max_words = 1000\n",
        "max_len = 150\n",
        "tok = Tokenizer(num_words=max_words)\n",
        "tok.fit_on_texts(X_train)\n",
        "sequences = tok.texts_to_sequences(X_train)\n"
      ],
      "metadata": {
        "id": "uLmpG6GPuHvT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def RNN():\n",
        "    inputs = Input(name='inputs',shape=[max_len])\n",
        "    layer = Embedding(max_words,50,input_length=max_len)(inputs)\n",
        "    layer = LSTM(64)(layer)\n",
        "    layer = Dense(256,name='FC1')(layer)\n",
        "    layer = Activation('relu')(layer)\n",
        "    layer = Dropout(0.5)(layer)\n",
        "    layer = Dense(1,name='out_layer')(layer)\n",
        "    layer = Activation('sigmoid')(layer)\n",
        "    model = Model(inputs=inputs,outputs=layer)\n",
        "    return model"
      ],
      "metadata": {
        "id": "R2obNB9xubiM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = RNN()\n",
        "model.summary()\n",
        "model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SbcwG0I0ujWs",
        "outputId": "1f194994-a9b4-431f-841e-bddc1ad3edb2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " inputs (InputLayer)         [(None, 150)]             0         \n",
            "                                                                 \n",
            " embedding (Embedding)       (None, 150, 50)           50000     \n",
            "                                                                 \n",
            " lstm (LSTM)                 (None, 64)                29440     \n",
            "                                                                 \n",
            " FC1 (Dense)                 (None, 256)               16640     \n",
            "                                                                 \n",
            " activation (Activation)     (None, 256)               0         \n",
            "                                                                 \n",
            " dropout (Dropout)           (None, 256)               0         \n",
            "                                                                 \n",
            " out_layer (Dense)           (None, 1)                 257       \n",
            "                                                                 \n",
            " activation_1 (Activation)   (None, 1)                 0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 96,337\n",
            "Trainable params: 96,337\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.columns"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PVX_L2ZY1Wi-",
        "outputId": "413eb61e-8939-4f63-efde-77d9050e8f05"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['v1', 'v2', 'Count'], dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 65
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data=df.rename(\n",
        "{\n",
        "    \"v1\":\"Category\",\n",
        "    \"v2\":\"Message\"\n",
        "},\n",
        "    axis=1\n",
        ")"
      ],
      "metadata": {
        "id": "zY-3FoIr12Oo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PtR27mAE2CxC",
        "outputId": "53f430ed-5e99-4d73-efcb-aa47a32d3c84"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 5572 entries, 1211 to 3623\n",
            "Data columns (total 3 columns):\n",
            " #   Column  Non-Null Count  Dtype \n",
            "---  ------  --------------  ----- \n",
            " 0   v1      5572 non-null   object\n",
            " 1   v2      5572 non-null   object\n",
            " 2   Count   5572 non-null   int64 \n",
            "dtypes: int64(1), object(2)\n",
            "memory usage: 174.1+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[\"Message Length\"]=data[\"Message\"].apply(len)"
      ],
      "metadata": {
        "id": "288h3yHZ2KTF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig=plt.figure(figsize=(12,8))\n",
        "sns.histplot(\n",
        "    x=data[\"Message Length\"],\n",
        "    hue=data[\"Category\"]\n",
        ")\n",
        "plt.title(\"ham & spam messege length comparision\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 513
        },
        "id": "tzukuS8H2OkW",
        "outputId": "053b1c0f-497c-4c11-ebf4-6e8ad1e644a2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 864x576 with 1 Axes>"
            ],
            "image/png": "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u+VI0qpxmIckSZJUkz3TkiRJUk2GaUmSJKmmfv3Qlre+9a05cuTI3i5DkiRJq7l77733ucwcsnx7vw7TI0eO5J577untMiRJkrSai4iZbbU7zEOSJEmqyTAtSZIk1WSYliRJkmrq12OmJUmS1P0WL15MS0sLixYt6u1SetygQYMYPnw4AwcO7NT6hmlJkiS9SUtLCxtttBEjR44kInq7nB6TmTz//PO0tLQwatSoTm3jMA9JkiS9yaJFi9hss83WqCANEBFsttlmXeqRN0xLkiRpBWtakG7V1eM2TEuSJKm2Z555hiOPPJJ3vOMd7LLLLuy///78z//8T5vrzps3j5/+9Kc9XGFzGaYlSZJUS2bykY98hL333pvHHnuMe++9l+985zs8++yzba7fU2F6yZIlTX+PVoZpSZIk1XLbbbcxcOBAPvnJTy5r22GHHdhpp53Yd9992XnnnRk9ejTXXHMNAKeccgqPPfYYO+64IyeffDIA3//+99l1113ZfvvtOe2005bt51vf+hbbbrste+21F0cddRQ/+MEPAHjggQfYfffd2X777fnIRz7Ciy++CMDee+/NSSedxJgxYzjjjDMYNWoUixcvBuCll15603x38m4ekiRJqmXq1KnssssuK7QPGjSIq6++mre85S0899xz7L777hx00EF897vfZerUqTzwwAMA3HzzzTz66KPcddddZCYHHXQQf/jDH1hvvfW46qqrePDBB1m8eDE777zzsvc55phj+MlPfsL73/9+/umf/olvfOMb/PCHPwTgtdde45577gFgxowZ/OY3v+Hggw/msssu45BDDun07e66wjAtSZKkbpWZfO1rX+MPf/gDAwYMYPbs2W0O/bj55pu5+eab2WmnnQBYuHAhjz76KAsWLGD8+PEMGjSIQYMGceCBBwIwf/585s2bx/vf/34AJkyYwGGHHbZsf0ccccSy6Y9//ON873vf4+CDD+aCCy7g3HPPbcqxGqYlSZJUy3vf+16uvPLKFdovueQS5s6dy7333svAgQMZOXJkm7eby0y++tWv8olPfOJN7a09zV21wQYbLJvec889mTFjBr/73e9YunQp2223Xa19roxjpiVJklTLPvvsw6uvvsqkSZOWtT300EPMnDmTzTffnIEDB3Lbbbcxc+ZMADbaaCMWLFiwbN2xY8dy/vnns3DhQgBmz57N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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ham_desc=data[data[\"Category\"]==\"ham\"][\"Message Length\"].describe()\n",
        "spam_desc=data[data[\"Category\"]==\"spam\"][\"Message Length\"].describe()\n",
        "\n",
        "print(\"Ham Messege Length Description:\\n\",ham_desc)\n",
        "print(\"************************************\")\n",
        "print(\"Spam Message Length Description:\\n\",spam_desc)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V9jbx0xO2TWx",
        "outputId": "759c7367-d67f-439f-9aa8-e2d2c8f95283"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ham Messege Length Description:\n",
            " count    4825.000000\n",
            "mean       71.023627\n",
            "std        58.016023\n",
            "min         2.000000\n",
            "25%        33.000000\n",
            "50%        52.000000\n",
            "75%        92.000000\n",
            "max       910.000000\n",
            "Name: Message Length, dtype: float64\n",
            "************************************\n",
            "Spam Message Length Description:\n",
            " count    747.000000\n",
            "mean     138.866131\n",
            "std       29.183082\n",
            "min       13.000000\n",
            "25%      132.500000\n",
            "50%      149.000000\n",
            "75%      157.000000\n",
            "max      224.000000\n",
            "Name: Message Length, dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.describe(include=\"all\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 394
        },
        "id": "-xeUsVN22XZh",
        "outputId": "68252490-c24f-4615-972f-24c5167ded3a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       Category                 Message   Count  Message Length\n",
              "count      5572                    5572  5572.0     5572.000000\n",
              "unique        2                    5169     NaN             NaN\n",
              "top         ham  Sorry, I'll call later     NaN             NaN\n",
              "freq       4825                      30     NaN             NaN\n",
              "mean        NaN                     NaN     0.0       80.118808\n",
              "std         NaN                     NaN     0.0       59.690841\n",
              "min         NaN                     NaN     0.0        2.000000\n",
              "25%         NaN                     NaN     0.0       36.000000\n",
              "50%         NaN                     NaN     0.0       61.000000\n",
              "75%         NaN                     NaN     0.0      121.000000\n",
              "max         NaN                     NaN     0.0      910.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-1b7fb4c8-2e09-40ec-a4e8-faa159c3688c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Category</th>\n",
              "      <th>Message</th>\n",
              "      <th>Count</th>\n",
              "      <th>Message Length</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>5572</td>\n",
              "      <td>5572</td>\n",
              "      <td>5572.0</td>\n",
              "      <td>5572.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>unique</th>\n",
              "      <td>2</td>\n",
              "      <td>5169</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>top</th>\n",
              "      <td>ham</td>\n",
              "      <td>Sorry, I'll call later</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>freq</th>\n",
              "      <td>4825</td>\n",
              "      <td>30</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.0</td>\n",
              "      <td>80.118808</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.0</td>\n",
              "      <td>59.690841</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.0</td>\n",
              "      <td>36.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.0</td>\n",
              "      <td>61.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.0</td>\n",
              "      <td>121.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.0</td>\n",
              "      <td>910.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-1b7fb4c8-2e09-40ec-a4e8-faa159c3688c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-1b7fb4c8-2e09-40ec-a4e8-faa159c3688c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-1b7fb4c8-2e09-40ec-a4e8-faa159c3688c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 73
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[\"Category\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BufFRx4i2cKt",
        "outputId": "b5a4bda3-79c9-48d2-9963-5b3711035634"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "ham     4825\n",
              "spam     747\n",
              "Name: Category, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 74
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(\n",
        "    data=data,\n",
        "    x=\"Category\"\n",
        ")\n",
        "plt.title(\"ham vs spam\")\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "wTWnsrlD2eAp",
        "outputId": "a04180fd-f098-49e3-a0dc-c98c25d344ae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ham_count=data[\"Category\"].value_counts()[0]\n",
        "spam_count=data[\"Category\"].value_counts()[1]\n",
        "\n",
        "total_count=data.shape[0]\n",
        "\n",
        "print(\"Ham contains:{:.2f}% of total data.\".format(ham_count/total_count*100))\n",
        "print(\"Spam contains:{:.2f}% of total data.\".format(spam_count/total_count*100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aZrZ9HMZ2k46",
        "outputId": "0baecbdb-d590-47b2-a2df-c43159700833"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ham contains:86.59% of total data.\n",
            "Spam contains:13.41% of total data.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#compute the length of majority & minority class\n",
        "minority_len=len(data[data[\"Category\"]==\"spam\"])\n",
        "majority_len=len(data[data[\"Category\"]==\"ham\"])\n",
        "\n",
        "#store the indices of majority and minority class\n",
        "minority_indices=data[data[\"Category\"]==\"spam\"].index\n",
        "majority_indices=data[data[\"Category\"]==\"ham\"].index\n",
        "\n",
        "#generate new majority indices from the total majority_indices\n",
        "#with size equal to minority class length so we obtain equivalent number of indices length\n",
        "random_majority_indices=np.random.choice(\n",
        "    majority_indices,\n",
        "    size=minority_len,\n",
        "    replace=False\n",
        ")\n",
        "\n",
        "#concatenate the two indices to obtain indices of new dataframe\n",
        "undersampled_indices=np.concatenate([minority_indices,random_majority_indices])\n",
        "\n",
        "#create df using new indices\n",
        "df=data.loc[undersampled_indices]\n",
        "\n",
        "#shuffle the sample\n",
        "df=df.sample(frac=1)\n",
        "\n",
        "#reset the index as its all mixed\n",
        "df=df.reset_index()\n",
        "\n",
        "#drop the older index\n",
        "df=df.drop(\n",
        "    columns=[\"index\"],\n",
        ")"
      ],
      "metadata": {
        "id": "Sip1u7bB2qVS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zWjLowZM2wzO",
        "outputId": "2dc6c8b2-f526-44d4-d43e-11a9e7a377cd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1494, 4)"
            ]
          },
          "metadata": {},
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df[\"Category\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LShbWuFD22hN",
        "outputId": "03848484-810e-4706-9157-dd129f176e29"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "spam    747\n",
              "ham     747\n",
              "Name: Category, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(\n",
        "    data=df,\n",
        "    x=\"Category\"\n",
        ")\n",
        "plt.title(\"ham vs spam\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "nbHcgFdW25l5",
        "outputId": "98ff8ad5-1b32-4944-866d-959c0226bcfe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "hp7UdIE_3EIR",
        "outputId": "185648ab-e4f3-4873-910c-99fc0c29a4e0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  Category                                            Message  Count  \\\n",
              "0     spam  Eerie Nokia tones 4u, rply TONE TITLE to 8007 ...      0   \n",
              "1      ham  That sucks. I'll go over so u can do my hair. ...      0   \n",
              "2      ham   says that he's quitting at least5times a day ...      0   \n",
              "3      ham  Hey. For me there is no leave on friday. Wait ...      0   \n",
              "4     spam  Please call our customer service representativ...      0   \n",
              "\n",
              "   Message Length  \n",
              "0             162  \n",
              "1              70  \n",
              "2             200  \n",
              "3              83  \n",
              "4             149  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-459f0d4a-e9ce-4372-9dfd-bf213e16b390\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Category</th>\n",
              "      <th>Message</th>\n",
              "      <th>Count</th>\n",
              "      <th>Message Length</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>spam</td>\n",
              "      <td>Eerie Nokia tones 4u, rply TONE TITLE to 8007 ...</td>\n",
              "      <td>0</td>\n",
              "      <td>162</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>That sucks. I'll go over so u can do my hair. ...</td>\n",
              "      <td>0</td>\n",
              "      <td>70</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>ham</td>\n",
              "      <td>says that he's quitting at least5times a day ...</td>\n",
              "      <td>0</td>\n",
              "      <td>200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>Hey. For me there is no leave on friday. Wait ...</td>\n",
              "      <td>0</td>\n",
              "      <td>83</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>spam</td>\n",
              "      <td>Please call our customer service representativ...</td>\n",
              "      <td>0</td>\n",
              "      <td>149</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-459f0d4a-e9ce-4372-9dfd-bf213e16b390')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-459f0d4a-e9ce-4372-9dfd-bf213e16b390 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-459f0d4a-e9ce-4372-9dfd-bf213e16b390');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 81
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df[\"Label\"]=df[\"Category\"].map(\n",
        "    {\n",
        "        \"ham\":0,\n",
        "        \"spam\":1\n",
        "    }\n",
        ")"
      ],
      "metadata": {
        "id": "-pnfaDR33Jvx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "sAlIyiEo3LV-",
        "outputId": "eb5c22ec-e50c-4eef-acd3-b4778a291ed5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  Category                                            Message  Count  \\\n",
              "0     spam  Eerie Nokia tones 4u, rply TONE TITLE to 8007 ...      0   \n",
              "1      ham  That sucks. I'll go over so u can do my hair. ...      0   \n",
              "2      ham   says that he's quitting at least5times a day ...      0   \n",
              "3      ham  Hey. For me there is no leave on friday. Wait ...      0   \n",
              "4     spam  Please call our customer service representativ...      0   \n",
              "\n",
              "   Message Length  Label  \n",
              "0             162      1  \n",
              "1              70      0  \n",
              "2             200      0  \n",
              "3              83      0  \n",
              "4             149      1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b374a99e-0a0b-42cb-89f0-b5f3dc908a48\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Category</th>\n",
              "      <th>Message</th>\n",
              "      <th>Count</th>\n",
              "      <th>Message Length</th>\n",
              "      <th>Label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>spam</td>\n",
              "      <td>Eerie Nokia tones 4u, rply TONE TITLE to 8007 ...</td>\n",
              "      <td>0</td>\n",
              "      <td>162</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>That sucks. I'll go over so u can do my hair. ...</td>\n",
              "      <td>0</td>\n",
              "      <td>70</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>ham</td>\n",
              "      <td>says that he's quitting at least5times a day ...</td>\n",
              "      <td>0</td>\n",
              "      <td>200</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>Hey. For me there is no leave on friday. Wait ...</td>\n",
              "      <td>0</td>\n",
              "      <td>83</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>spam</td>\n",
              "      <td>Please call our customer service representativ...</td>\n",
              "      <td>0</td>\n",
              "      <td>149</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b374a99e-0a0b-42cb-89f0-b5f3dc908a48')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b374a99e-0a0b-42cb-89f0-b5f3dc908a48 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b374a99e-0a0b-42cb-89f0-b5f3dc908a48');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 83
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import PorterStemmer\n",
        "\n",
        "stemmer=PorterStemmer()"
      ],
      "metadata": {
        "id": "Imy7m5FP3R3J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#declare empty list to store tokenized message\n",
        "corpus=[]\n",
        "\n",
        "#iterate through the df[\"Message\"]\n",
        "for message in df[\"Message\"]:\n",
        "    \n",
        "    #replace every special characters, numbers etc.. with whitespace of message\n",
        "    #It will help retain only letter/alphabets\n",
        "    message=re.sub(\"[^a-zA-Z]\",\" \",message)\n",
        "    \n",
        "    #convert every letters to its lowercase\n",
        "    message=message.lower()\n",
        "    \n",
        "    #split the word into individual word list\n",
        "    message=message.split()\n",
        "    \n",
        "   "
      ],
      "metadata": {
        "id": "ICLU4ZFk3TmW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.preprocessing.text import one_hot\n",
        "vocab_size=10000\n",
        "\n",
        "oneHot_doc=[one_hot(words,n=vocab_size)\n",
        "           for words in corpus\n",
        "           ]"
      ],
      "metadata": {
        "id": "ej79ocye3wWQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df[\"Message Length\"].describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lg5KwjR_3zbR",
        "outputId": "8928a22b-7b66-44a4-9221-4694de1c4835"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "count    1494.000000\n",
              "mean      103.384873\n",
              "std        55.635473\n",
              "min         2.000000\n",
              "25%        48.000000\n",
              "50%       115.000000\n",
              "75%       152.750000\n",
              "max       408.000000\n",
              "Name: Message Length, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 92
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fig=plt.figure(figsize=(12,8))\n",
        "sns.kdeplot(\n",
        "    x=df[\"Message Length\"],\n",
        "    hue=df[\"Category\"]\n",
        ")\n",
        "plt.title(\"ham & spam messege length comparision\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 513
        },
        "id": "dtW0HVKy32xp",
        "outputId": "554ad0b9-7f9b-4df3-b90a-f6aa103a87a6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 864x576 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "sentence_len=200\n",
        "embedded_doc=pad_sequences(\n",
        "    oneHot_doc,\n",
        "    maxlen=sentence_len,\n",
        "    padding=\"pre\"\n",
        ")"
      ],
      "metadata": {
        "id": "OAa6ngzA362J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "extract_features=pd.DataFrame(\n",
        "    data=embedded_doc\n",
        ")\n",
        "target=df[\"Label\"]"
      ],
      "metadata": {
        "id": "J9MtqSG-3_EO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_final=pd.concat([extract_features,target],axis=1)"
      ],
      "metadata": {
        "id": "nvDVPaJ74C_W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_final.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 235
        },
        "id": "nh43o9_c4F-X",
        "outputId": "910eb5c8-5140-401c-ce11-1d40163407f1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "    0   1   2   3   4   5   6   7   8   9  ...  191  192  193  194  195  196  \\\n",
              "0 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN   \n",
              "1 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN   \n",
              "2 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN   \n",
              "3 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN   \n",
              "4 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN   \n",
              "\n",
              "   197  198  199  Label  \n",
              "0  NaN  NaN  NaN      1  \n",
              "1  NaN  NaN  NaN      0  \n",
              "2  NaN  NaN  NaN      0  \n",
              "3  NaN  NaN  NaN      0  \n",
              "4  NaN  NaN  NaN      1  \n",
              "\n",
              "[5 rows x 201 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-21cc39ea-cc0e-4fcc-82fb-f5cbebc3a806\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "      <th>2</th>\n",
              "      <th>3</th>\n",
              "      <th>4</th>\n",
              "      <th>5</th>\n",
              "      <th>6</th>\n",
              "      <th>7</th>\n",
              "      <th>8</th>\n",
              "      <th>9</th>\n",
              "      <th>...</th>\n",
              "      <th>191</th>\n",
              "      <th>192</th>\n",
              "      <th>193</th>\n",
              "      <th>194</th>\n",
              "      <th>195</th>\n",
              "      <th>196</th>\n",
              "      <th>197</th>\n",
              "      <th>198</th>\n",
              "      <th>199</th>\n",
              "      <th>Label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 201 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-21cc39ea-cc0e-4fcc-82fb-f5cbebc3a806')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-21cc39ea-cc0e-4fcc-82fb-f5cbebc3a806 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-21cc39ea-cc0e-4fcc-82fb-f5cbebc3a806');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 97
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X=df_final.drop(\"Label\",axis=1)\n",
        "y=df_final[\"Label\"]"
      ],
      "metadata": {
        "id": "b6YlHCm74IY0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "metadata": {
        "id": "FMhs9pQk4Oyr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_trainval,X_test,y_trainval,y_test=train_test_split(\n",
        "    X,\n",
        "    y,\n",
        "    random_state=42,\n",
        "    test_size=0.15\n",
        ")"
      ],
      "metadata": {
        "id": "q2U8X9T-4QRN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_val,y_train,y_val=train_test_split(\n",
        "    X_trainval,\n",
        "    y_trainval,\n",
        "    random_state=42,\n",
        "    test_size=0.15\n",
        ")"
      ],
      "metadata": {
        "id": "JTTAHkiC4TcS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = RNN()\n",
        "model.summary()\n",
        "model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UxtNcb3h9NQ_",
        "outputId": "8054930d-e823-40a8-a85b-ab2beacc633d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model_3\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " inputs (InputLayer)         [(None, 150)]             0         \n",
            "                                                                 \n",
            " embedding_4 (Embedding)     (None, 150, 50)           50000     \n",
            "                                                                 \n",
            " lstm_4 (LSTM)               (None, 64)                29440     \n",
            "                                                                 \n",
            " FC1 (Dense)                 (None, 256)               16640     \n",
            "                                                                 \n",
            " activation_6 (Activation)   (None, 256)               0         \n",
            "                                                                 \n",
            " dropout_3 (Dropout)         (None, 256)               0         \n",
            "                                                                 \n",
            " out_layer (Dense)           (None, 1)                 257       \n",
            "                                                                 \n",
            " activation_7 (Activation)   (None, 1)                 0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 96,337\n",
            "Trainable params: 96,337\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    }
  ]
}