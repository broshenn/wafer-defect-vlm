# Codex D36逐图AI复核

新一轮AI图像事实检查，不替代人工gold；不把未观察到主要错误解释为真实正确率；不改旧模型复核或历史成绩。

每行均有原图与原答哈希；方向不适用、未回答和答错分开。

|编号|形态|位置|遗漏|无依据断言|方向适用/判断|图像证据|
|---|---|---|---|---|---|---|
|R01|PARTIAL|PARTIAL|CONTRADICTED|SUPPORTED|True/NO_ANSWER|中右至右下存在较明显红色块，同时边缘与内部有散点；只说中心致密团簇遗漏外侧块。|
|R02|SUPPORTED|SUPPORTED|PARTIAL|SUPPORTED|False/NO_ANSWER|中心附近有相连红色团块并混有全图散点；中心团簇有依据，范围较小的限定略强。|
|R03|CONTRADICTED|PARTIAL|CONTRADICTED|PARTIAL|True/NO_ANSWER|上半部有宽而相连的横向/分支红色带；稀疏随机点的caption遗漏主要连续结构。|
|R04|PARTIAL|SUPPORTED|PARTIAL|SUPPORTED|unknown/UNCERTAIN|中部有断续弧形结构，环形只能部分支持；12点聚集不足以唯一确定主方向。|
|R05|CONTRADICTED|CONTRADICTED|CONTRADICTED|NEEDS_REVIEW|True/NO_ANSWER|左侧至左下有大面积连续红区，不能只写全局随机混合；46.5%未给测量依据，本次不替它证明数值。|
|R06|PARTIAL|SUPPORTED|PARTIAL|SUPPORTED|True/NO_ANSWER|下缘红色弧带与上缘红色聚集并存；边缘位置成立，无明显方向的表述过于笼统。|
|R07|PARTIAL|SUPPORTED|PARTIAL|SUPPORTED|False/NO_ANSWER|外缘红色带较明确，局部断续；内部散点较多，少量与连续的表述需限定。|
|R08|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|False/NO_ANSWER|红色窄带沿外缘近连续分布，内部少量散点；主体描述与图相符。|
|R09|PARTIAL|PARTIAL|PARTIAL|SUPPORTED|True/SUPPORTED|中心向下有明显纵向细长红色连接块，6点方向有依据；仅团簇未充分表达线状延伸。|
|R10|PARTIAL|CONTRADICTED|PARTIAL|SUPPORTED|True/PARTIAL|全图非常稀疏，较明显小块接近上缘偏左；中部限定不符，12点只近似反映上方位置。|
|R11|SUPPORTED|PARTIAL|SUPPORTED|PARTIAL|False/NO_ANSWER|全图红色占主导，绿色残留也在内部；仅边缘残留正常点的限定不符。|
|R12|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|False/NO_ANSWER|大范围红区，右侧及局部残留绿色；广泛致密的描述有图像依据。|
|R13|PARTIAL|SUPPORTED|PARTIAL|PARTIAL|False/NO_ANSWER|红色碎块分布全图且不少；随机散布有依据，稀疏限定不够稳妥。|
|R14|CONTRADICTED|PARTIAL|CONTRADICTED|PARTIAL|False/NO_ANSWER|中央存在明显相连红色块，外围仍有散点；只写稀疏随机遗漏核心结构。|
|R15|SUPPORTED|CONTRADICTED|SUPPORTED|SUPPORTED|True/PARTIAL|细长红线位于上右外侧、近1至2点；划痕成立，主要位置应偏外缘而非中部。|
|R16|PARTIAL|PARTIAL|PARTIAL|SUPPORTED|True/CONTRADICTED|细长断续红色结构主要横跨中下部，约左下到右侧；1点主方向缺乏支持。|
|R17|PARTIAL|CONTRADICTED|PARTIAL|PARTIAL|False/NO_ANSWER|红点分布跨中心至外侧，图内还有黑色空洞；不宜只限定为中心局部。|
|R18|PARTIAL|SUPPORTED|SUPPORTED|PARTIAL|False/NO_ANSWER|红色块分散覆盖全图；无单一主结构较合理，少量的强度限定需谨慎。|
|R19|SUPPORTED|SUPPORTED|PARTIAL|PARTIAL|False/NO_ANSWER|外缘宽红色环带明显，内部散点也较多；环形准确，少量内部散点略弱化内部信息。|
|R20|PARTIAL|SUPPORTED|CONTRADICTED|PARTIAL|True/PARTIAL|有中下部纵向红色带，也有向右上延伸及横向分支；只写6点一条线遗漏多方向结构。|
|R21|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|False/NO_ANSWER|边缘有断续红色环带，内部为散点；没有显著整体主方向。|
|R22|PARTIAL|SUPPORTED|CONTRADICTED|PARTIAL|False/NO_ANSWER|弧带和局部块状聚集并存，左中部较大红块明显；所有方向连续环带的说法过强。|
|R23|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|False/NO_ANSWER|中心致密相连红色团块与外围散点并存；中心团簇描述较吻合。|
|R24|CONTRADICTED|CONTRADICTED|CONTRADICTED|PARTIAL|unknown/UNCERTAIN|较粗红块散布，较明显聚集靠左上外侧；图上不支持单个中心致密连续团簇。|
|R25|PARTIAL|SUPPORTED|PARTIAL|PARTIAL|False/NO_ANSWER|红色块在全图广泛出现且较密，随机散布部分成立；稀疏与无聚集都过强。|
|R26|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|False/NO_ANSWER|全图散布离散红色块，没有突出主形态；少量、分散的描述基本吻合。|
|R27|SUPPORTED|SUPPORTED|PARTIAL|SUPPORTED|False/NO_ANSWER|中央周围存在断续弧形红色结构，外围伴散点；环形有依据，但结构不够完整。|
|R28|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|False/NO_ANSWER|几乎全部红色，正常绿色残留在下缘等极少区域；近全覆盖有直接依据。|
|R29|PARTIAL|PARTIAL|PARTIAL|PARTIAL|True/NO_ANSWER|中心和下半部大红色区域相连，上半部保留明显绿色；近乎全覆盖、无方向均过强。|
|R30|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|False/NO_ANSWER|红色块分散于全图，无突出连续主模式；caption基本吻合。|
|R31|PARTIAL|SUPPORTED|PARTIAL|PARTIAL|False/NO_ANSWER|全图红色碎块及较大相连块不少；随机可部分支持，但稀疏限定不稳妥。|
|R32|PARTIAL|SUPPORTED|PARTIAL|PARTIAL|False/NO_ANSWER|分散红色块覆盖全图，局部相连；随机点状部分成立，稀疏限定需谨慎。|
|R33|PARTIAL|PARTIAL|PARTIAL|PARTIAL|False/NO_ANSWER|外缘局部聚集和内部散点并存；只写稀疏边缘点弱化局部聚集与内部信息。|
|R34|PARTIAL|SUPPORTED|PARTIAL|SUPPORTED|True/SUPPORTED|中心偏下右有较明显弯曲相连红块，4至5点区域有依据；团簇表述未充分表达弧状分支。|
|R35|SUPPORTED|SUPPORTED|PARTIAL|SUPPORTED|True/SUPPORTED|右下外侧明显局部红色聚集并伴散点；4点附近近似合理，另有下侧小结构未充分描述。|
|R36|SUPPORTED|SUPPORTED|SUPPORTED|SUPPORTED|True/SUPPORTED|中心偏上红色相连团块明显；中心、近12点方向的主要描述有依据。|
