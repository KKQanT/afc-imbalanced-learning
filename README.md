# afc-imbalanced-learning
A simple implementation of Adaptive Feature-Space Conformal Transformationfor Imbalanced-Data Learning.

## Core concept
In [1], the idea of inceasing the separability (margin) of classes for kernel SVM, can be achieved by magnify the spatial resolution in the region around the boundary in the hyperspace, the non-linear serface where $\textbf{x}$ has been already mapped to $\phi(\textbf{x})$ using kernel tricks. 

Given $K(\textbf{x}, \textbf{x}') = \langle \phi(\textbf{x}), \phi(\textbf{x}')\rangle$ is a kernel function corresponding to the non-linear mapping function $\phi(\textbf{x})$ for a kernel SVM. We can employ a conformal transformation of kernel:

$$
\begin{aligned}
 \tilde{K}(\textbf{x}, \textbf{x}') = D(\textbf{x})D(\textbf{x}')K(\textbf{x}, \textbf{x}')
\end{aligned}
$$

to enlarge the distance near the boundary in non-linear surface in higher dimension by choosing appropriate $D(\textbf{x})$ that has a high value when $\textbf{x}$ is close to the boundary and small when it is far away from the boundary.

In [1], it has been suggested that $D(\textbf{x})$ can be chosen as:

$$
\begin{aligned}
 D(\textbf{x}) = \sum_{\textbf{x}_k \epsilon SV} e^{ -\frac{1}{\tau_k^{2}}||\textbf{x} - \textbf{x}_k||^{2}}
\end{aligned}
$$

where 

$$
\begin{aligned}
 \tau_k^{2} = \frac{1}{M} \sum_{\textbf{x}_s \epsilon SV_k} || \textbf{x}_s - \textbf{x}_k||^{2} ||
\end{aligned}
$$

where $SV_{k}$ denotes a set of $M$ support vectors $\textbf{x}_{s}$ that are nearest to support vector $\textbf{x}_k$

In this implementation, we use the $\tau_k^{2}$ proposed in [2]:

$$
\begin{aligned}
 \tau_k^{2} = AVG_{\textbf{x}_s \epsilon \{\textbf{x}_s \epsilon SV | \ || \phi{(\textbf{x}_s)} - \phi{(\textbf{x}_k)} ||^2 < M, \ y_s \ne y_k, \}} (|| \phi{(\textbf{x}_s)} - \phi{(\textbf{x}_k)} ||^2)
\end{aligned}
$$

where $M$ is the mean distance squared (in hyperspace) of the nearest and the farthest support vector from $\phi(\textbf{x}_k)$. It has been mentioned in [2] that by setting $\tau_k$ like this will take into account the spatial distribution of the support vectors in hyperspace.

Note that given kernel function, $K(\textbf{x}, \textbf{x}')$ is known but the mapping function $\phi(\textbf{x})$ is unknown, we can still calculate the distance in hyperspace using Kernel trick

$$
\begin{aligned}
    | \phi{(\textbf{x}_s)} - \phi{(\textbf{x}_k)} ||^2 = K(\textbf{x}_s, \textbf{x}_s) +  K(\textbf{x}_k, \textbf{x}_k) - 2K(\textbf{x}_s, \textbf{x}_k)
\end{aligned}
$$

**Dealing with Imbalaned**
<br/>
It has been stated in [2] that the $\tau_k^{2}$ will be scaled with a larger factor $\eta_p$ if $\textbf{x}_k$ belong to minority class and will be scaled down with a factor $\eta_n$ to address imbalance issue. The paper suggest that we should choose $\eta_p$ and $\eta_n$ proportional to the skew of support vectors. In this version of implementation, I set $\eta_p$ = 1 and $\eta_n = \frac{|SV^{+}|}{|SV^{-}|}$ for now.

## Example Usage

```
from afc_imbalanced_learning import AFSCTSvm

afc_svm = AFSCTSvm()
afc_svm.fit(X_train, y_train)
y_pred = afc_svm.predict(X_test)
```

what `.fit(X_train, y_train)` does is it train kernel svm with laplacian kernel $K(\textbf{x}, \textbf{x}') = e^{-\gamma|\textbf{x} - \textbf{x}'|}$ as used in [2], then it estimates the location of boundary by extracting support vectors and it then calculate $\tau_k$ for every support vectors and then calculate $D(\textbf{x})$ to use for conformal transformation where we'll obtain $\tilde{K}(\textbf{x}, \textbf{x})$ to train our new SVM. We'll then use the new improved Kernel SVM to predict when `.predict` is called.

you can use your own custom kernel function by parse it to kernel parameter
```
from sklearn.metrics.pairwise import rbf_kernel
afc_svm = AFSCTSvm(kernel=rbf_kernel)
afc_svm.fit(X_train, y_train)
```

Note that in this implementation, cost-sensitive svm will be used by default

## References
<a id="1">[1]</a>
Wu, Si and Shun‐ichi Amari. “Conformal Transformation of Kernel Functions: A Data-Dependent Way to Improve Support Vector Machine Classifiers.” Neural Processing Letters 15 (2002): 59-67.
<br/>
<a id="2">[2]</a> 
Wu, Gang & Chang, Edward. (2003). Adaptive Feature-Space Conformal Transformation for Imbalanced-Data Learning. Proceedings, Twentieth International Conference on Machine Learning. 2. 816-823. 
