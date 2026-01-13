# (1) 相互情報量のフィッシャー情報量積分表現を利用した相互情報量推定法の提案
## arXiv:2510.05496 https://www.arxiv.org/abs/2510.05496
Tadashi Wadayama
Mutual Information Estimation via Score-to-Fisher Bridge for Nonlinear Gaussian Noise Channels

We present a numerical method to evaluate mutual information (MI) in nonlinear Gaussian noise channels by using denoising score matching (DSM) learning for estimating the score function of channel output. Via de Bruijn's identity, Fisher information estimated from the learned score function yields accurate estimates of MI through a Fisher integral representation for a variety of priors and channel nonlinearities. In this work, we propose a comprehensive theoretical foundation for the Score-to-Fisher bridge methodology, along with practical guidelines for its implementation. We also conduct extensive validation experiments, comparing our approach with closed-form solutions and a kernel density estimation baseline. The results of our numerical experiments demonstrate that the proposed method is both practical and efficient for MI estimation in nonlinear Gaussian noise channels. Additionally, we discuss the theoretical connections between our score-based framework and thermodynamic concepts, such as partition function estimation and optimal transport.

本論文では、非線形ガウス雑音通信路における相互情報量（MI）を評価するための数値的手法を提案する。この手法では、通信路出力のスコア関数を推定するために、デノイジングスコアマッチング（DSM）学習を用いる。de Bruijnの恒等式を介して、学習されたスコア関数から推定されるFisher情報量は、様々な事前分布や通信路の非線形性に対して、Fisher積分表現を通じてMIの正確な推定値を与える。本研究では、Score-to-Fisher bridge手法の包括的な理論的基盤を提案するとともに、その実装に関する実践的なガイドラインを示す。また、閉形式解やカーネル密度推定によるベースラインと比較する広範な検証実験を行う。数値実験の結果は、提案手法が非線形ガウス雑音通信路におけるMI推定において実用的かつ効率的であることを示している。さらに、スコアベースのフレームワークと分配関数推定や最適輸送などの熱力学的概念との理論的関連についても議論する。

---

# (2) 情報勾配：相互情報量の勾配のスコア関数表現に基づく相互情報量最大化
## arXiv:2510.20179 https://arxiv.org/abs/2510.20179
Tadashi Wadayama
Information Gradient for Nonlinear Gaussian Channel with Applications to Task-Oriented Communication

We propose a gradient-based framework for optimizing parametric nonlinear Gaussian channels via mutual information maximization. Leveraging the score-to-Fisher bridge (SFB) methodology, we derive a computationally tractable formula for the information gradient that is the gradient of mutual information with respect to the parameters of the nonlinear front-end. Our formula expresses this gradient in terms of two key components: the score function of the marginal output distribution, which can be learned via denoising score matching (DSM), and the Jacobian of the front-end function, which is handled efficiently using the vector-Jacobian product (VJP) within automatic differentiation frameworks. This enables practical parameter optimization through gradient ascent. Furthermore, we extend this framework to task-oriented scenarios, deriving gradients for both task-specific mutual information, where a task variable depends on the channel input, and the information bottleneck (IB) objective. A key advantage of our approach is that it facilitates end-to-end optimization of the nonlinear front-end without requiring explicit computation on the output distribution. Extensive experimental validation confirms the correctness of our information gradient formula against analytical solutions and demonstrates its effectiveness in optimizing both linear and nonlinear channels toward their objectives.

本論文では、相互情報量の最大化を通じてパラメトリックな非線形ガウス通信路を最適化するための勾配ベースのフレームワークを提案する。Score-to-Fisher bridge（SFB）手法を活用し、非線形フロントエンドのパラメータに関する相互情報量の勾配である情報勾配について、計算可能な公式を導出する。提案する公式は、この勾配を2つの主要な構成要素で表現する：デノイジングスコアマッチング（DSM）により学習可能な周辺出力分布のスコア関数と、自動微分フレームワーク内でベクトル・ヤコビアン積（VJP）を用いて効率的に扱えるフロントエンド関数のヤコビアンである。これにより、勾配上昇法による実用的なパラメータ最適化が可能となる。さらに、タスク指向シナリオへの拡張として、タスク変数が通信路入力に依存するタスク固有相互情報量と、情報ボトルネック（IB）目的関数の両方に対する勾配を導出する。本アプローチの重要な利点は、出力分布に対する明示的な計算を必要とせずに、非線形フロントエンドのエンドツーエンド最適化を可能にする点である。広範な実験的検証により、情報勾配公式の正しさが解析解との比較で確認され、線形・非線形通信路の両方においてそれぞれの目的に向けた最適化の有効性が実証されている。

---

# (3) 情報勾配：有向非巡回グラフへの拡張
## arXiv:2601.01789 https://arxiv.org/abs/2601.01789

Tadashi Wadayama
Information Gradient for Directed Acyclic Graphs: A Score-based Framework for End-to-End Mutual Information Maximization

This paper presents a general framework for end-to-end mutual information maximization in communication and sensing systems represented by stochastic directed acyclic graphs (DAGs). We derive a unified formula for the (mutual) information gradient with respect to arbitrary internal parameters, utilizing marginal and conditional score functions. We demonstrate that this gradient can be efficiently computed using vector-Jacobian products (VJP) within standard automatic differentiation frameworks, enabling the optimization of complex networks under global resource constraints. Numerical experiments on both linear multipath DAGs and nonlinear channels validate the proposed framework; the results confirm that the estimator, utilizing score functions learned via denoising score matching, accurately reproduces ground-truth gradients and successfully maximizes end-to-end mutual information. Beyond maximization, we extend our score-based framework to a novel unsupervised paradigm: digital twin calibration via Fisher divergence minimization.

 本論文では、確率的有向非巡回グラフ（DAG）で表現される通信およびセンシングシステムにおけるエンドツーエンド相互情報量最大化のための一般的なフレームワークを提示する。周辺スコア関数と条件付きスコア関数を活用し、任意の内部パラメータに関する（相互）情報勾配の統一的な公式を導出する。この勾配が標準的な自動微分フレームワーク内でベクトル・ヤコビアン積（VJP）を用いて効率的に計算可能であることを示し、大域的な資源制約の下での複雑なネットワークの最適化を可能にする。線形マルチパスDAGと非線形通信路の両方に対する数値実験により、提案フレームワークを検証する。その結果、デノイジングスコアマッチングにより学習されたスコア関数を用いた推定器が、真の勾配を正確に再現し、エンドツーエンド相互情報量の最大化に成功することが確認される。最大化を超えて、スコアベースのフレームワークを新しい教師なしパラダイムへ拡張する：Fisherダイバージェンス最小化によるデジタルツインキャリブレーションである。

---

# (4)スコア関数とフィッシャー情報量を利用したVAMP(逆問題ソルバー)
## arXiv:2601.07095 https://arxiv.org/abs/2601.07095

Tadashi Wadayama and Takumi Takahashi
Score-Based VAMP  with Fisher-Information-Based Onsager Correction

We propose score-based VAMP (SC-VAMP), a vari-
ant of vector approximate message passing (VAMP) in which the
Onsager correction is expressed and computed via conditional
Fisher information, thereby enabling a Jacobian-free imple-
mentation. Using learned score functions, SC-VAMP constructs
nonlinear minimum mean square error (MMSE) estimators
through Tweedie’s formula and derives the corresponding On-
sager terms from the score-norm statistics, avoiding the need for
analytical derivatives of the prior or likelihood. When combined
with random orthogonal/unitary mixing to mitigate non-ideal,
structured or correlated sensing settings, the proposed frame-
work extends VAMP to complex black-box inference problems
where explicit modeling is intractable. Finally, by leveraging the
entropic central limit theorem (CLT), we provide an information-
theoretic perspective on the Gaussian approximation underlying
state evolution (SE), offering insight into the decoupling principle
beyond idealized independent and identically distributed (i.i.d.)
settings, including nonlinear regimes.

本稿では、ベクトル近似メッセージパッシング（VAMP）の一変種であるスコアベースVAMP（SC-VAMP）を提案する。SC-VAMPでは、Onsager補正を条件付きFisher情報量を用いて表現・計算することで、ヤコビアンを必要としない実装を可能にしている。学習されたスコア関数を用いて、SC-VAMPはTweedieの公式を通じて非線形の最小平均二乗誤差（MMSE）推定器を構成し、対応するOnsager項をスコアノルム統計量から導出することで、事前分布や尤度の解析的な微分を不要としている。ランダム直交/ユニタリ混合と組み合わせることで、非理想的、構造化された、あるいは相関のあるセンシング設定を軽減し、提案フレームワークは明示的なモデル化が困難な複雑なブラックボックス推論問題へとVAMPを拡張する。最後に、エントロピー中心極限定理（CLT）を活用することで、状態発展（SE）の基礎となるガウス近似に対する情報理論的な視点を提供し、非線形領域を含む理想化された独立同分布（i.i.d.）設定を超えた分離原理への洞察を与える。
