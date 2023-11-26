
models = {}


Linear_Regression = {
    "video_links": [
        {
            "title": "***Linear Regression with One Variable***",
            "url": "https://www.youtube.com/watch?v=W46UTQ_JDPk",
        },
        {
            "title": "***Linear Regression with Multiple Variables***",
            "url": "https://www.youtube.com/watch?v=UVCFaaEBnTE",
        },
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Linear_regression",
        },
        {
            "title": "***Scikit-Learn***",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#",
        },
    ],
}



Logistic_Regression = {
    "video_links": [
        {
            "title": "***Logistic Regression in 3 Minutes***",
            "url": "https://www.youtube.com/watch?v=EKm0spFxFG4",
        },

        {
            "title": "***Logistic Regression***",
            "url": "https://www.youtube.com/watch?v=hjrYrynGWGA",
        },
      
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Logistic_regression",
        },
        {
            "title": "***Scikit-Learn***",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        },
    ],
}

Linear_SVC = {
    "video_links": [
        {
            "title": "***Support Vector Machine***",
            "url": "https://www.youtube.com/watch?v=uV5TnFc7eaE",
        },
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Support_vector_machine",
        },
        {
            "title": "***Scikit-Learn***",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html",
        },
    ],
}

Linear_Alebra = {
    "video_links": [
        {
            "title": """***Linear Algebra video series by 3Blue1Brown***
            This series is a great resource for learning Linear Algebra. It is highly recommended to watch the entire series.   
            """,
            "url": "https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab",
        },
    ],
    "hyperlinks": [
        {
            "title": "***More about 3Blue1Brown***",
            "url": "https://www.3blue1brown.com",
        },
    ],

}
Calculus = {
    "video_links": [
        {
            "title": """***Calculus video series by 3Blue1Brown***
            This series are great resources for learning Calculus. It is highly recommended to watch the entire series.
            """,
            "url": "https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr",
        },
    ],
    "hyperlinks": [
        {
            "title": """***More about 3Blue1Brown***""",
            "url": "https://www.3blue1brown.com",
        },
    ],

}

Decision_Tree = {
    "video_links": [
        {
            "title": "***Decision Tree***",
            "url": "https://www.youtube.com/watch?v=AdhG64NF76E&t=4s",
        },
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Decision_tree",
        },
        {
            "title": "***Scikit-Learn***",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
        },
    ],
} 

Ensemable_Methods = {
    "video_links": [
        {
            "title": "***Ensemble Methods***",
            "url": "https://www.youtube.com/watch?v=wr9gUr-eWdA",
        },
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Ensemble_learning",
        },
    ],

}

Neural_Networks = {
    "video_links": [
        {
            "title": "***Neural Networks***",
            "url": "https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF",
        },
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Artificial_neural_network",
        },
    ],

}

Res_Nets = {
    "video_links": [
        {
            "title": "***Residual Networks***",
            "url": "https://www.youtube.com/watch?v=ZILIbUvp5lk",
        },
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Residual_neural_network",
        },
    ],

}
Data_Processing_Normalization = {
    "video_links": [
        {
            "title": "***Normalization***",
            "url": "https://www.youtube.com/watch?v=FDCfw-YqWTE",
        },
    ],
    "hyperlinks": [
        {
            "title": "***Wikipedia***",
            "url": "https://en.wikipedia.org/wiki/Normalization_(statistics)",
        },
        {
            "title": "***Scikit-Learn***",
            "url": "https://scikit-learn.org/stable/modules/preprocessing.html#normalization",
        },
    ],

}

models.update({"Linear Algebra": Linear_Alebra})
models.update({"Calculus": Calculus})
models.update({"Data Processing & Normalization": Data_Processing_Normalization})
models.update({"Linear Regression": Linear_Regression})
models.update({"Linear Support Vector ": Linear_SVC})
models.update({"Decision Tree": Decision_Tree})
models.update({"Logistic Regression": Logistic_Regression})
models.update({"Ensemble Methods": Ensemable_Methods})
models.update({"Neural Networks": Neural_Networks})
models.update({"Residual Networks": Res_Nets})

