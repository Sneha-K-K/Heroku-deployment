{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.7.6"
    },
    "colab": {
      "name": "Linear_Regression_(1)to submit.ipynb",
      "provenance": [],
      "collapsed_sections": [
        "AzYKorxH0w67",
        "BEJUfkWG0w7P"
      ]
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "s1aByW200w6R"
      },
      "source": [
        "#  <center><u>LINEAR REGRESSION<u><center>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n2WGTXS-0w6Z"
      },
      "source": [
        "<p style='text-align: right;'> Total points =51</p>\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JOR1QLNy0w6b"
      },
      "source": [
        "###  Linear Regression\n",
        "\n",
        "\n",
        "Linear Regression is a statistical technique which is used to find the linear relationship between dependent and one or more independent variables. This technique is applicable for Supervised learning Regression problems where we try to predict a continuous variable.\n",
        "\n",
        "\n",
        "Linear Regression can be further classified into two types – Simple and Multiple Linear Regression. It is the simplest form of Linear Regression where we fit a straight line to the data.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "posmQtq-0w6d"
      },
      "source": [
        "###  Simple Linear Regression (SLR)\n",
        "\n",
        "Simple Linear Regression (or SLR) is the simplest model in machine learning. It models the linear relationship between the independent and dependent variables. \n",
        "\n",
        "This assignment is based on the TV and Sales data .\n",
        "There is one independent or input variable which represents the TV data and is denoted by X. Similarly, there is one dependent or output variable which represents the Sales and is denoted by y. We want to build a linear relationship between these variables. This linear relationship can be modelled by mathematical equation of the form:-\n",
        "\t\t\t\t \n",
        "                 \n",
        "                 Y = β0   + β1*X    -------------   (1)\n",
        "                 \n",
        "\n",
        "In this equation, X and Y are called independent and dependent variables respectively,\n",
        "\n",
        "β1 is the coefficient for independent variable and\n",
        "\n",
        "β0 is the constant term.\n",
        "\n",
        "β0 and β1 are called parameters of the model.\n",
        " \n",
        "\n",
        "\n",
        "For simplicity, we can compare the above equation with the basic line equation of the form:-\n",
        " \n",
        "                   y = ax + b       ----------------- (2)\n",
        "\n",
        "We can see that \n",
        "\n",
        "slope of the line is given by, a =  β1,  and\n",
        "\n",
        "intercept of the line by b =  β0. \n",
        "\n",
        "\n",
        "In this Simple Linear Regression model, we want to fit a line which estimates the linear relationship between X and Y. So, the question of fitting reduces to estimating the parameters of the model β0 and β1. \n",
        "\n",
        " \n",
        "\n",
        "## Ordinary Least Square Method\n",
        "\n",
        "The TV and Sales data are given by X and y respectively. We can draw a scatter plot between X and y which shows the relationship between them.\n",
        "\n",
        " \n",
        "\n",
        "Now, our task is to find a line which best fits this scatter plot. This line will help us to predict the value of any Target variable for any given Feature variable. This line is called **Regression line**. \n",
        "\n",
        "\n",
        "We can define an error function for any line. Then, the regression line is the one which minimizes the error function. Such an error function is also called a **Cost function**. \n",
        "\n",
        "By below chart you might understand more clearly\n",
        "\n",
        "![image.png](attachment:image.png)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xkjaWXyE0w6f"
      },
      "source": [
        "Understanding the Data\n",
        "Let's start with the following steps:\n",
        "\n",
        "1. Importing data using the pandas library\n",
        "2. Understanding the structure of the data"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h4Q0Exxa0w6h"
      },
      "source": [
        "<p style='text-align: right;'> 2*2=4 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cmWywAgK0w6i"
      },
      "source": [
        "# Import necessary libraries\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "%matplotlib inline\n",
        "# import %matplotlib inline to visualise in the notebook"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2bRmQwzv0w6m"
      },
      "source": [
        "## About the dataset\n",
        "\n",
        "Let's import data from the following url:-\n",
        "\n",
        "https://www.kaggle.com/ashydv/advertising-dataset\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_kCfRI2c0w6n"
      },
      "source": [
        "<p style='text-align: right;'> 2*6 = 12 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lm5pYg3N0w6p",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "ecb964d9-85d6-4268-e5ed-00cd96ee9b67"
      },
      "source": [
        "# Import the data as df\n",
        "\n",
        "df=pd.read_csv('advertising.csv')\n",
        "df.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      TV  Radio  Newspaper  Sales\n",
              "0  230.1   37.8       69.2   22.1\n",
              "1   44.5   39.3       45.1   10.4\n",
              "2   17.2   45.9       69.3   12.0\n",
              "3  151.5   41.3       58.5   16.5\n",
              "4  180.8   10.8       58.4   17.9"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a15c0841-91bb-4400-976c-b5dea49ac076\">\n",
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
              "      <th>TV</th>\n",
              "      <th>Radio</th>\n",
              "      <th>Newspaper</th>\n",
              "      <th>Sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>230.1</td>\n",
              "      <td>37.8</td>\n",
              "      <td>69.2</td>\n",
              "      <td>22.1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>44.5</td>\n",
              "      <td>39.3</td>\n",
              "      <td>45.1</td>\n",
              "      <td>10.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>17.2</td>\n",
              "      <td>45.9</td>\n",
              "      <td>69.3</td>\n",
              "      <td>12.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>151.5</td>\n",
              "      <td>41.3</td>\n",
              "      <td>58.5</td>\n",
              "      <td>16.5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>180.8</td>\n",
              "      <td>10.8</td>\n",
              "      <td>58.4</td>\n",
              "      <td>17.9</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a15c0841-91bb-4400-976c-b5dea49ac076')\"\n",
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
              "          document.querySelector('#df-a15c0841-91bb-4400-976c-b5dea49ac076 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-a15c0841-91bb-4400-976c-b5dea49ac076');\n",
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
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RnftQiHP0w6q",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "outputId": "1b9c578b-4de5-493a-c69c-b1385fb76b29"
      },
      "source": [
        "#drop radio and newspaper column from df\n",
        "df.drop(['Radio','Newspaper'],axis=1)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        TV  Sales\n",
              "0    230.1   22.1\n",
              "1     44.5   10.4\n",
              "2     17.2   12.0\n",
              "3    151.5   16.5\n",
              "4    180.8   17.9\n",
              "..     ...    ...\n",
              "195   38.2    7.6\n",
              "196   94.2   14.0\n",
              "197  177.0   14.8\n",
              "198  283.6   25.5\n",
              "199  232.1   18.4\n",
              "\n",
              "[200 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-034b9438-1a9b-45cb-9b30-35da49ea2ade\">\n",
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
              "      <th>TV</th>\n",
              "      <th>Sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>230.1</td>\n",
              "      <td>22.1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>44.5</td>\n",
              "      <td>10.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>17.2</td>\n",
              "      <td>12.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>151.5</td>\n",
              "      <td>16.5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>180.8</td>\n",
              "      <td>17.9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>195</th>\n",
              "      <td>38.2</td>\n",
              "      <td>7.6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>196</th>\n",
              "      <td>94.2</td>\n",
              "      <td>14.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>197</th>\n",
              "      <td>177.0</td>\n",
              "      <td>14.8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>198</th>\n",
              "      <td>283.6</td>\n",
              "      <td>25.5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>199</th>\n",
              "      <td>232.1</td>\n",
              "      <td>18.4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>200 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-034b9438-1a9b-45cb-9b30-35da49ea2ade')\"\n",
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
              "          document.querySelector('#df-034b9438-1a9b-45cb-9b30-35da49ea2ade button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-034b9438-1a9b-45cb-9b30-35da49ea2ade');\n",
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
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uKCM1DSL0w6r"
      },
      "source": [
        "### pandas shape attribute\n",
        "\n",
        "The shape attribute of the pandas dataframe gives the dimensions of the dataframe."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ObibRuJm0w6s",
        "outputId": "bc1c3bc7-6863-4ca6-966a-fde00a57997e",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# View the dimensions of df\n",
        "\n",
        "df.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(200, 4)"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3A2JSfxz0w6u"
      },
      "source": [
        "### pandas head() method\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "z5tLbWkv0w6u",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "8937cb6b-cfa0-4e00-eb97-c69ce37ac0ca"
      },
      "source": [
        "# View the top 5 rows of df\n",
        "\n",
        "df.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      TV  Radio  Newspaper  Sales\n",
              "0  230.1   37.8       69.2   22.1\n",
              "1   44.5   39.3       45.1   10.4\n",
              "2   17.2   45.9       69.3   12.0\n",
              "3  151.5   41.3       58.5   16.5\n",
              "4  180.8   10.8       58.4   17.9"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-13183f15-5da7-4a7f-a185-86b8f2b5fb17\">\n",
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
              "      <th>TV</th>\n",
              "      <th>Radio</th>\n",
              "      <th>Newspaper</th>\n",
              "      <th>Sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>230.1</td>\n",
              "      <td>37.8</td>\n",
              "      <td>69.2</td>\n",
              "      <td>22.1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>44.5</td>\n",
              "      <td>39.3</td>\n",
              "      <td>45.1</td>\n",
              "      <td>10.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>17.2</td>\n",
              "      <td>45.9</td>\n",
              "      <td>69.3</td>\n",
              "      <td>12.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>151.5</td>\n",
              "      <td>41.3</td>\n",
              "      <td>58.5</td>\n",
              "      <td>16.5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>180.8</td>\n",
              "      <td>10.8</td>\n",
              "      <td>58.4</td>\n",
              "      <td>17.9</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-13183f15-5da7-4a7f-a185-86b8f2b5fb17')\"\n",
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
              "          document.querySelector('#df-13183f15-5da7-4a7f-a185-86b8f2b5fb17 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-13183f15-5da7-4a7f-a185-86b8f2b5fb17');\n",
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
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gpRypyAQ0w6v"
      },
      "source": [
        "### pandas info() method"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BpgfBwNg0w6w",
        "outputId": "62647fdf-fef3-42c0-d88f-59e18434eea0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# View dataframe summary\n",
        "\n",
        "df.info()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 200 entries, 0 to 199\n",
            "Data columns (total 4 columns):\n",
            " #   Column     Non-Null Count  Dtype  \n",
            "---  ------     --------------  -----  \n",
            " 0   TV         200 non-null    float64\n",
            " 1   Radio      200 non-null    float64\n",
            " 2   Newspaper  200 non-null    float64\n",
            " 3   Sales      200 non-null    float64\n",
            "dtypes: float64(4)\n",
            "memory usage: 6.4 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Yg6gE8Tz0w6x"
      },
      "source": [
        "### pandas describe() method"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PRi2wqCv0w6x",
        "outputId": "2a2967f6-c489-439a-e19c-b3c34e610420",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        }
      },
      "source": [
        "# View descriptive statistics\n",
        "df.describe()\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "               TV       Radio   Newspaper       Sales\n",
              "count  200.000000  200.000000  200.000000  200.000000\n",
              "mean   147.042500   23.264000   30.554000   15.130500\n",
              "std     85.854236   14.846809   21.778621    5.283892\n",
              "min      0.700000    0.000000    0.300000    1.600000\n",
              "25%     74.375000    9.975000   12.750000   11.000000\n",
              "50%    149.750000   22.900000   25.750000   16.000000\n",
              "75%    218.825000   36.525000   45.100000   19.050000\n",
              "max    296.400000   49.600000  114.000000   27.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-580f9865-e7a0-46a0-8510-63a42bd07d7d\">\n",
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
              "      <th>TV</th>\n",
              "      <th>Radio</th>\n",
              "      <th>Newspaper</th>\n",
              "      <th>Sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>200.000000</td>\n",
              "      <td>200.000000</td>\n",
              "      <td>200.000000</td>\n",
              "      <td>200.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>147.042500</td>\n",
              "      <td>23.264000</td>\n",
              "      <td>30.554000</td>\n",
              "      <td>15.130500</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>85.854236</td>\n",
              "      <td>14.846809</td>\n",
              "      <td>21.778621</td>\n",
              "      <td>5.283892</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.700000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.300000</td>\n",
              "      <td>1.600000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>74.375000</td>\n",
              "      <td>9.975000</td>\n",
              "      <td>12.750000</td>\n",
              "      <td>11.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>149.750000</td>\n",
              "      <td>22.900000</td>\n",
              "      <td>25.750000</td>\n",
              "      <td>16.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>218.825000</td>\n",
              "      <td>36.525000</td>\n",
              "      <td>45.100000</td>\n",
              "      <td>19.050000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>296.400000</td>\n",
              "      <td>49.600000</td>\n",
              "      <td>114.000000</td>\n",
              "      <td>27.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-580f9865-e7a0-46a0-8510-63a42bd07d7d')\"\n",
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
              "          document.querySelector('#df-580f9865-e7a0-46a0-8510-63a42bd07d7d button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-580f9865-e7a0-46a0-8510-63a42bd07d7d');\n",
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
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7G418lBi0w6y"
      },
      "source": [
        "## Independent and Dependent Variables\n",
        "\n",
        "\n",
        "### Independent variable\n",
        "\n",
        "Independent variable is also called Input variable and is denoted by X. In practical applications, independent variable is also called Feature variable or Predictor variable. We can denote it as:-\n",
        "\n",
        "Independent or Input variable (X) = Feature variable = Predictor variable \n",
        "\n",
        "\n",
        "### Dependent variable\n",
        "\n",
        "Dependent variable is also called Output variable and is denoted by y. \n",
        "\n",
        "Dependent variable is also called Target variable or Response variable. It can be denoted it as follows:-\n",
        "\n",
        "Dependent or Output variable (y) = Target variable = Response variable\n",
        "\n",
        "\n",
        "\n",
        "Reference :https://youtu.be/XZLQwZ0hs0A\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nzOL4oU50w6z"
      },
      "source": [
        "**bold text**<p style='text-align: right;'> 2 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kIRoDrhX0w6z"
      },
      "source": [
        "# Declare feature variable and target variable\n",
        "# TV and Sales data values are given by X and y respectively.\n",
        "# Values attribute of pandas dataframe returns the numpy arrays.\n",
        "X=df.loc[:,'TV']\n",
        "Y=df.loc[:,'Sales']\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3OFVqnw50w60"
      },
      "source": [
        "## Visual exploratory data analysis\n",
        "\n",
        "Visualize the relationship between X and y by plotting a scatterplot between X and y."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tPwMrIRJ16Wo"
      },
      "source": [
        "Reference : https://youtu.be/WvbLLNZyvrM"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9sdsO7Ec0w61"
      },
      "source": [
        "<p style='text-align: right;'> 2 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hyBZWZ590w61",
        "outputId": "acb31b6d-e1ca-429a-a3b2-fc0707d0d16b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        }
      },
      "source": [
        "# Plot scatter plot between X and y\n",
        "\n",
        "plt.scatter(X,Y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7f450ffd1fd0>"
            ]
          },
          "metadata": {},
          "execution_count": 6
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
      "cell_type": "markdown",
      "metadata": {
        "id": "FwpDJPUb0w62"
      },
      "source": [
        "Hey buddy! did you notice ? the above graph shows some sort of relationship between sales and TV. Don't you think this shows positive linear relation? i.e when As TV's value increases sales increases ans same is vise-versa."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XtMij8fT0w62"
      },
      "source": [
        "# Visualising Data Using Seaborn\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xelLuLmL0w63"
      },
      "source": [
        "<p style='text-align: right;'> 2*2=4 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kVFCSqbX0w63",
        "outputId": "78504f53-04ed-4cd4-a1ed-0acb851a6404",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 708
        }
      },
      "source": [
        "# Visualise the relationship between the features and the response using scatterplots\n",
        "import seaborn as sns\n",
        "sns.scatterplot(X,Y)\n",
        "sns.pairplot(df[['TV','Sales']])\n",
        "\n",
        "# plot a pairplot also for df\n",
        "\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  FutureWarning\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<seaborn.axisgrid.PairGrid at 0x7f450ff76f10>"
            ]
          },
          "metadata": {},
          "execution_count": 7
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
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 360x360 with 6 Axes>"
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
      "cell_type": "markdown",
      "metadata": {
        "id": "JmVFYdgN0w64"
      },
      "source": [
        "Ohoo! We can see very well that you have done good practice of Visualisation in your EDA assignment. Anyways the above graph also shows positive linear relation between both TV and Sales."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "m8blklSa0w64"
      },
      "source": [
        "## Checking dimensions of X and y\n",
        "\n",
        "We need to check the dimensions of X and y to make sure they are in right format for Scikit-Learn API. \n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cB0p3CnQ3Onc"
      },
      "source": [
        "refrence - https://youtu.be/2un1b7EEBwc"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HSkxTq0_0w64"
      },
      "source": [
        "<p style='text-align: right;'> 2points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oXyXjoU40w65",
        "outputId": "58ab23fc-35b8-4fba-beeb-13ff242d3d62",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# Print the dimensions of X and y\n",
        "print(X.shape,Y.shape)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(200,) (200,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kzJHXYXU0w65"
      },
      "source": [
        "## Reshaping X and y\n",
        "\n",
        "Since we are working with only one feature variable, so we need to reshape using Numpy reshape() method.\n",
        "\n",
        "E.g, If you have an array of shape (3,2) then reshaping it with (-1, 1), then the array will get reshaped in such a way that the resulting array has only 1 column and this is only possible by having 6 rows, hence, (6,1)\n",
        "\n",
        "You have seen the above example. Now you smarty! try reshaping on your data."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2n7gzHKw0w65"
      },
      "source": [
        "<p style='text-align: right;'> 2*2 = 4 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "woSt4Hie0w66"
      },
      "source": [
        "# Reshape X and y\n",
        "X=np.reshape(df[['TV']],(-1,1))\n",
        "Y=np.reshape(df[['Sales']],(-1,1))\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ijkBI1pd0w66",
        "outputId": "1f924525-4d90-4376-8b4f-24167d5794b5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# Print the dimensions of X and y after reshaping\n",
        "print(X.shape,Y.shape)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(200, 1) (200, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AzYKorxH0w67"
      },
      "source": [
        "Cool right!\n",
        "\n",
        "## Difference in dimensions of X and y after reshaping\n",
        "\n",
        "\n",
        "Hey Intellipants! You can see the difference in diminsions of X and y before and after reshaping.\n",
        "\n",
        "It is essential in this case because getting the feature and target variable right is an important precursor to model building."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eu2rwYp30w67"
      },
      "source": [
        "# Performing Simple Linear Regression"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YcHNU5A70w68"
      },
      "source": [
        "Equation of linear regression<br>\n",
        "$y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$\n",
        "\n",
        "-  $y$ is the response\n",
        "-  $c$ is the intercept\n",
        "-  $m_1$ is the coefficient for the first feature\n",
        "-  $m_n$ is the coefficient for the nth feature<br>\n",
        "\n",
        "In our case:\n",
        "\n",
        "$y = c + m_1 \\times TV$\n",
        "\n",
        "The $m$ values are called the model **coefficients** or **model parameters**.\n",
        "\n",
        "Reference - https://youtu.be/Jx_I4GLXLys"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Vbjau4Jr0w68"
      },
      "source": [
        "## Mechanics of the model\n",
        "\n",
        "Hey! before you read further, it is good to understand the generic structure of modeling using the scikit-learn library. Broadly, the steps to build any model can be divided as follows: \n",
        "\n",
        "Split the dataset into two sets – the training set and the test set. Then, instantiate the regressor lm and fit it on the training set with the fit method. \n",
        "\n",
        "In this step, the model learned the relationships between the training data (X_train, y_train). \n",
        "\n",
        "Oh Yeah! Now the model is ready to make predictions on the test data (X_test). Hence, predict on the test data using the predict method. \n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Dn0U6gfR0w68"
      },
      "source": [
        "The steps are as follow:"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qdAcHz9h0w69"
      },
      "source": [
        "## Train test split\n",
        "\n",
        "\n",
        "Split the dataset into two sets namely - train set and test set.\n",
        "\n",
        "The model learn the relationships from the training data and predict on test data.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "v6TMScHC0w69"
      },
      "source": [
        "Hey Smartypants!! It's absolutely fine if you didn't understand the theory well! We are here to help you make comfortable with all the concepts slowly as we proceeds towards our upcoming assignments.\n",
        "\n",
        "No fear when AI_4_All is here :)\n",
        "\n",
        "<p style='text-align: right;'> 2+2+3=7 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zrAhs1ZP0w69"
      },
      "source": [
        "# import train_test_split module\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# Split X and y into training and test data sets with test_size=0.3 and random_state=42\n",
        "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wY_yTydL0w6-",
        "outputId": "a822efef-9453-480a-8250-ca7eb65574db",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# print shapes of X_train,y_train, X_test, y_test\n",
        "\n",
        "print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(140, 1) (60, 1) (140, 1) (60, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "64vAbhuo0w6_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "de22f27d-ede3-4f00-bbdd-4e7f4c06a35b"
      },
      "source": [
        "# import LinearRegression module\n",
        "\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "# Instantiate the linear regression object lm\n",
        "lm=LinearRegression()\n",
        "\n",
        "\n",
        "# Fit and train the model using training data sets\n",
        "lm.fit(X_train,Y_train)\n",
        "\n",
        "\n",
        "# Predict on the test data\n",
        "y_pred=lm.predict(X_test)\n",
        "y_pred\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[16.16109107],\n",
              "       [17.92129084],\n",
              "       [23.26769201],\n",
              "       [ 7.84812269],\n",
              "       [19.28667945],\n",
              "       [11.32465432],\n",
              "       [19.09475735],\n",
              "       [ 9.94829874],\n",
              "       [19.4018327 ],\n",
              "       [16.8081427 ],\n",
              "       [ 8.93385339],\n",
              "       [10.28827502],\n",
              "       [20.0653348 ],\n",
              "       [ 7.50266292],\n",
              "       [14.85602084],\n",
              "       [16.53945177],\n",
              "       [ 7.6068492 ],\n",
              "       [18.04192759],\n",
              "       [11.3356213 ],\n",
              "       [20.22435596],\n",
              "       [19.79116038],\n",
              "       [10.92435967],\n",
              "       [ 9.29028013],\n",
              "       [20.96462689],\n",
              "       [10.99016153],\n",
              "       [10.14022083],\n",
              "       [18.91380224],\n",
              "       [14.84505386],\n",
              "       [11.98815642],\n",
              "       [ 7.66716757],\n",
              "       [18.16256433],\n",
              "       [11.00112851],\n",
              "       [18.11321294],\n",
              "       [ 8.13326408],\n",
              "       [22.59870643],\n",
              "       [20.26822387],\n",
              "       [ 9.85507944],\n",
              "       [22.21486224],\n",
              "       [13.63320293],\n",
              "       [ 8.71451385],\n",
              "       [13.56740107],\n",
              "       [16.91232898],\n",
              "       [ 9.56993804],\n",
              "       [10.62276781],\n",
              "       [19.48956852],\n",
              "       [ 9.30124711],\n",
              "       [11.07789734],\n",
              "       [15.28373293],\n",
              "       [12.94228339],\n",
              "       [11.39045618],\n",
              "       [11.49464246],\n",
              "       [16.44074898],\n",
              "       [ 7.68361804],\n",
              "       [ 7.63426664],\n",
              "       [11.39593967],\n",
              "       [14.30218851],\n",
              "       [11.23143502],\n",
              "       [23.09222038],\n",
              "       [ 8.28131827],\n",
              "       [18.04192759]])"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Qk4E0qNe0w6_"
      },
      "source": [
        "## Model slope and intercept term\n",
        "\n",
        "The model slope is given by lm.coef_ and model intercept term is given by lm.intercept_. \n",
        "\n",
        "for example. if the estimated model slope and intercept values are 1.60509347 and  -11.16003616.\n",
        "\n",
        "So, the equation of the fitted regression line will be:-\n",
        "\n",
        "y = 1.60509347 * x - 11.16003616  \n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y1dgV7Cj0w6_"
      },
      "source": [
        "<p style='text-align: right;'> 2 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yIiIxFkM0w7A",
        "outputId": "e7b2c136-ce50-4342-9081-3aa0b4f03e61",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# Compute model slope and intercept\n",
        "a=lm.coef_\n",
        "b=lm.intercept_\n",
        "print(\"slope:\",a,\"intercept:\",b)\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "slope: [[0.05483488]] intercept: [7.20655455]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "P-yLbbB90w7B"
      },
      "source": [
        "# So comment below, our fitted regression line here is ?\n",
        "\n",
        "#y=0.054x+7.206\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yR7zn9ch0w7B"
      },
      "source": [
        "That is our linear model. Wohoo! Awesome job done! \n",
        "\n",
        "## Making predictions\n",
        "\n",
        "\n",
        "To make prediction, on an individual TV value, \n",
        "\n",
        "\n",
        "\t\tlm.predict(Xi)\n",
        "        \n",
        "\n",
        "where Xi is the TV data value of the ith observation.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vNRFt3hj0w7B"
      },
      "source": [
        "<p style='text-align: right;'> 2 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Jxlt9qZS0w7C",
        "outputId": "6920101b-6049-4a2c-9307-84195b1200ba",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# Predicting Sales values on first five 5 TV  datasets only\n",
        "\n",
        "lm.predict(X_test[0:5])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[16.16109107],\n",
              "       [17.92129084],\n",
              "       [23.26769201],\n",
              "       [ 7.84812269],\n",
              "       [19.28667945]])"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ve_h8Xkp0w7C"
      },
      "source": [
        "We know that you can also do prediction for all values of TV available in our dataset\n",
        "\n",
        "Can you show it now?\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "72qNKVrN0w7D",
        "outputId": "668e22cb-f569-4b6c-a0b4-3e893e1b29f3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# prediction for all X present in the dataset\n",
        "lm.predict(X_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[16.16109107],\n",
              "       [17.92129084],\n",
              "       [23.26769201],\n",
              "       [ 7.84812269],\n",
              "       [19.28667945],\n",
              "       [11.32465432],\n",
              "       [19.09475735],\n",
              "       [ 9.94829874],\n",
              "       [19.4018327 ],\n",
              "       [16.8081427 ],\n",
              "       [ 8.93385339],\n",
              "       [10.28827502],\n",
              "       [20.0653348 ],\n",
              "       [ 7.50266292],\n",
              "       [14.85602084],\n",
              "       [16.53945177],\n",
              "       [ 7.6068492 ],\n",
              "       [18.04192759],\n",
              "       [11.3356213 ],\n",
              "       [20.22435596],\n",
              "       [19.79116038],\n",
              "       [10.92435967],\n",
              "       [ 9.29028013],\n",
              "       [20.96462689],\n",
              "       [10.99016153],\n",
              "       [10.14022083],\n",
              "       [18.91380224],\n",
              "       [14.84505386],\n",
              "       [11.98815642],\n",
              "       [ 7.66716757],\n",
              "       [18.16256433],\n",
              "       [11.00112851],\n",
              "       [18.11321294],\n",
              "       [ 8.13326408],\n",
              "       [22.59870643],\n",
              "       [20.26822387],\n",
              "       [ 9.85507944],\n",
              "       [22.21486224],\n",
              "       [13.63320293],\n",
              "       [ 8.71451385],\n",
              "       [13.56740107],\n",
              "       [16.91232898],\n",
              "       [ 9.56993804],\n",
              "       [10.62276781],\n",
              "       [19.48956852],\n",
              "       [ 9.30124711],\n",
              "       [11.07789734],\n",
              "       [15.28373293],\n",
              "       [12.94228339],\n",
              "       [11.39045618],\n",
              "       [11.49464246],\n",
              "       [16.44074898],\n",
              "       [ 7.68361804],\n",
              "       [ 7.63426664],\n",
              "       [11.39593967],\n",
              "       [14.30218851],\n",
              "       [11.23143502],\n",
              "       [23.09222038],\n",
              "       [ 8.28131827],\n",
              "       [18.04192759]])"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "30wlIjQe0w7D"
      },
      "source": [
        "## Regression metrics for model performance\n",
        "\n",
        "\n",
        "Now, it is the time to evaluate model performance. \n",
        "\n",
        "For regression problems, there are two ways to compute the model performance. They are RMSE (Root Mean Square Error) and R-Squared Value. These are explained below:-  \n",
        "\n",
        "\n",
        "### RMSE\n",
        "\n",
        "    RMSE is the standard deviation of the residuals. So, RMSE gives us the standard deviation of the unexplained variance by the model. It can be calculated by taking square root of Mean Squared Error.\n",
        "    RMSE is an absolute measure of fit. It gives us how spread the residuals are, given by the standard deviation of the residuals. The more concentrated the data is around the regression line, the lower the residuals and hence lower the standard deviation of residuals. It results in lower values of RMSE. So, lower values of RMSE indicate better fit of data. \n",
        "\n",
        "Formula:\n",
        "![image.png](attachment:image.png)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "063BZSjx0w7E"
      },
      "source": [
        "### R-Squared\n",
        "\n",
        "    (R2) Correlation explains the strength of the relationship between an independent and dependent variable,whereas R-square explains to what extent the variance of one variable explains the variance of the second variable. Hence It may also be known as the coefficient of determination.\n",
        "    So, if the R2 of a model is 0.50, then approximately half of the observed variation can be explained by the model's inputs.\n",
        "    In general, the higher the R2 Score value, the better the model fits the data. Usually, its value ranges from 0 to 1. So, we want its value to be as close to 1. Its value can become negative if our model is wrong.\n",
        "\n",
        "Fomula:\n",
        "\n",
        "![image.png](attachment:image.png)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qNLjtnKa0w7E"
      },
      "source": [
        "<p style='text-align: right;'> 2*2 = 4 points</p>\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uBVkLw9a0w7F"
      },
      "source": [
        "Reference:\n",
        "https://youtu.be/PnOyLeekPVE"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wPizGGd40w7F",
        "outputId": "d6a66828-fa37-4ca3-fd37-5e8f6efcd4b0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# import mean_squared_error module\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "# Calculate and print Root Mean Square Error(RMSE)\n",
        "mse=mean_squared_error(Y_test,y_pred)\n",
        "print(mse)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5.179525402166653\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OkBaULeR0w7G",
        "outputId": "13be05b2-d5bf-498b-c6d3-1e16c5923d6b",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# import r2_score module\n",
        "\n",
        "from sklearn.metrics import r2_score\n",
        "# Calculate and print r2_score\n",
        "r2=r2_score(Y_test,y_pred)\n",
        "print(r2)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.814855389208679\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ejj56c2l0w7G"
      },
      "source": [
        "## Interpretation and Conclusion\n",
        "\n",
        "\n",
        "The RMSE value has been found to be  2.2759. It means the standard deviation for our prediction is  2.2759. which is quite less. Sometimes we can also expect the RMSE to be less than  2.2759. So, the model is good fit to the data. \n",
        "\n",
        "\n",
        "In business decisions, the benchmark for the R2 score value is 0.7. It means if R2 score value >= 0.7, then the model is good enough to deploy on unseen data whereas if R2 score value < 0.7, then the model is not good enough to deploy. Our R2 score value has been found to be  0.8149. It means that this model explains  81.49 % of the variance in our dependent variable. So, the R2 score value confirms that the model is good enough to deploy because it provides good fit to the data.\n",
        "\n",
        "Wohoo! Really good job done!"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DHqNSZCa0w7G"
      },
      "source": [
        "<p style='text-align: right;'> 2 points</p>\n",
        "\n",
        "Reference: https://www.youtube.com/watch?v=b0L47BeklTE"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kwl1ogBP0w7H",
        "outputId": "1d76c8a2-2aaf-4d3d-e1f3-b73cc9f8457b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        }
      },
      "source": [
        "# Plot the Regression Line between X and Y as shown in below output.\n",
        "plt.plot(X_test,y_pred,color='g',label=\"Linear Regression\")\n",
        "plt.scatter(X_test,Y_test,color='b',label=\"actual test data\")\n",
        "plt.legend()\n",
        "plt.show()\n",
        "\n"
      ],
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
      "cell_type": "markdown",
      "metadata": {
        "id": "Ge242Pn40w7H"
      },
      "source": [
        "As you can see above, the regression line fits the data quite well. Wow!"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fhx6QCSt0w7I"
      },
      "source": [
        "## Residual analysis\n",
        "\n",
        "Reference:https://youtu.be/OrWCtouG5jo\n",
        " \n",
        "\n",
        "A linear regression model may not represent the data appropriately. The model may be a poor fit to the data. So, we should validate our model by defining and examining residual plots.\n",
        "\n",
        "The difference between the observed value of the dependent variable (y) and the predicted value (ŷi) is called the residual and is denoted by e or error. The scatter-plot of these residuals is called residual plot.\n",
        "\n",
        "If the data points in a residual plot are randomly dispersed around horizontal axis and an approximate zero residual mean, a linear regression model may be appropriate for the data. Otherwise a non-linear model may be more appropriate.\n",
        "\n",
        "If we take a look at the generated ‘Residual errors’ plot, we can clearly see that the train data plot pattern is non-random. Same is the case with the test data plot pattern.\n",
        "So, it suggests a better-fit for a non-linear model. \n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HJYS1qE30w7I"
      },
      "source": [
        "<p style='text-align: right;'> 2 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "spZn5khg0w7I",
        "outputId": "804e1eff-2492-42f9-816a-4439539a153c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        }
      },
      "source": [
        "# Plotting residual errors\n",
        "Y_train_pred=lm.predict(X_train)\n",
        "train_residual=Y_train_pred-Y_train\n",
        "Y_test_pred=lm.predict(X_test)\n",
        "test_residual=Y_test_pred-Y_test\n",
        "plt.scatter(Y_train_pred,train_residual,color='c',label='Train data')\n",
        "plt.scatter(Y_test_pred,test_residual,color='purple',label='Test data')\n",
        "plt.legend()\n",
        "plt.show()\n",
        "\n"
      ],
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
      "cell_type": "markdown",
      "metadata": {
        "id": "OQ2eoTvZ0w7J"
      },
      "source": [
        "## Checking for Overfitting and Underfitting\n",
        "\n",
        "\n",
        "We will see training set score and test set score.\n",
        "\n",
        "You can excpect the training set score to be 0.7996, which is averagely good. So, the model learned the relationships quite appropriately from the training data. Thus, the model performs good on the test data as test score will be  0.8149. It is a clear sign of good fit/ balanced fit. Hence, we can validated our finding that the linear regression model provides good fit to the data. \n",
        "\n",
        "\n",
        "**Underfitting**: Your model is underfitting the training data when the model performs poorly on the training data. This is because the model is unable to capture the relationship between the input examples (often called X) and the target values (often called Y). \n",
        "\n",
        "**Overfitting**: Your model is overfitting your training data when you see that the model performs well on the training data but does not perform well on the evaluation data. This is because the model is memorizing the data it has seen and is unable to generalize to unseen examples.\n",
        "\n",
        "You see the difference visually as below:\n",
        "\n",
        "![image.png](attachment:image.png)\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "dKNQx5qkFwD2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DmnBhNft0w7J"
      },
      "source": [
        "<p style='text-align: right;'> 2 points</p>\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "h5Xwwzgq0w7J",
        "outputId": "bbc5a883-a83a-441d-dd82-1cd53298d26f",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# Checking for Overfitting or Underfitting the data by calculation score using score function.\n",
        "print(\"Training set score:\",lm.score(X_train,Y_train))\n",
        "print(\"Test set score:\",lm.score(X_test,Y_test))\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training set score: 0.799626928219267\n",
            "Test set score: 0.814855389208679\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle\n",
        "pickle.dump(lm,open('model.pkl','wb'))"
      ],
      "metadata": {
        "id": "6Iqb3aIuEzWF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cDh4j7yd0w7K"
      },
      "source": [
        "As you can see above that you have your simple linear model with train and test score close to each other, i.e your model is not overfitting or underfitting.\n",
        "But before we jump into any conclusion on this, it is always better to do cross validation, which will be introduced to you in your next assignment."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BEJUfkWG0w7P"
      },
      "source": [
        "## Simple Linear Regression - Model Assumptions\n",
        "\n",
        "Reference:- https://www.youtube.com/watch?v=rw84t7QU2O0\n",
        "\n",
        "The Linear Regression Model is based on several assumptions which are listed below:-\n",
        "\n",
        "i.\tLinear relationship\n",
        "ii.\tMultivariate normality\n",
        "iii.\tNo or little multicollinearity\n",
        "iv.\tNo auto-correlation\n",
        "v.\tHomoscedasticity\n",
        "\n",
        "\n",
        "### i.\tLinear relationship\n",
        "\n",
        "\n",
        "The relationship between response and feature variables should be linear. This linear relationship assumption can be tested by plotting a scatter-plot between response and feature variables.\n",
        "\n",
        "\n",
        "### ii.\tMultivariate normality\n",
        "\n",
        "The linear regression model requires all variables to be multivariate normal. A multivariate normal distribution means a vector in multiple normally distributed variables, where any linear combination of the variables is also normally distributed.\n",
        "\n",
        "\n",
        "### iii.\tNo or little multicollinearity\n",
        "\n",
        "It is assumed that there is little or no multicollinearity in the data. Multicollinearity occurs when the features (or independent variables) are highly correlated.\n",
        "\n",
        "\n",
        "### iv.\tNo auto-correlation\n",
        "\n",
        "Also, it is assumed that there is little or no auto-correlation in the data. Autocorrelation occurs when the residual errors are not independent from each other.\n",
        "\n",
        "\n",
        "### v.\tHomoscedasticity\n",
        "\n",
        "Homoscedasticity describes a situation in which the error term (that is, the noise in the model) is the same across all values of the independent variables. It means the residuals are same across the regression line. It can be checked by looking at scatter plot.\n"
      ]
    }
  ]
}