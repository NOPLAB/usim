# 津田沼: 形状最適化による窓枠・エッジ回収の次期計画

作成日: 2026-10-16
状態: 2026-10-16に局所実験・段階方式の実装と比較を実施。
結果: `runs/tsudanuma-isaac-local/edge-refinement/window-geometry-57/RESULTS.md`。
主窓は大幅改善、各軸非悪化・新領域・strict型検査は未達。
完全な反復結合最適化や全域展開には進めていない。以下は実行前に固定した計画を保持する。
対象: 現在の窓領域で、回収できた位置を維持しながら誤線・重複を減らす。

## 結論

**汎用メッシュを完成させてから全ての辺を拾う方式ではなく、
有限の面と桟の形状を点群に当てはめ、形状選択と配置を最適化する方式を優先する。**
エッジは選ばれた形状の関係・観測境界から取り出す。
点群は捨てず、最終的な検証の根拠として残す。

有力な研究上の系譜はGlobFitとRAPTERである。[R1][R2]
ただし以下は、それらの忠実再現ではなく、今回の窓に適した小規模比較実験の設計。
津田沼での改善効果は未実測で、最適化を導入すれば解決するとは断定しない。

## 誰の何を改善するか

ユーザーは実Isaacで元点と線を比較している。現状でも配置すべき位置は概ね拾えているが、
多数の斜線・重複が見通しを悪くしている。目指すのは次の状態。

| 目指す状態 | 現状との差 | 対応 |
| --- | --- | --- |
| 真の縦桟・横桟を残す | 回収率向上後の誤線除去で再欠落する危険 | 非悪化ゲート |
| 同じ桟は少数の安定した形状で説明する | 一本の部材に多数の軸仮説 | 点の所属と形状数を同時選択 |
| 斜めの偽線が横切らない | 別部材から点支持を借りても近接検査に通る | 面・断面の持続と競合説明 |
| 欠測と実開口を勝手に閉じない | 閉曲面再構成は補完を含む | 開いた有限形状と未知領域 |
| 細部が判別不能なら保留する | 単一面から厚みや裏面が決まらない | 識別可能性と不確実性を出力 |
| 見た目と採点で同じ結果を使う | 249本選別と194本表示が混在 | 共通の支持監査を全方式へ適用 |

## 既知の証拠と再利用

パスは全て`usim`ルートからの相対パス。

- `runs/tsudanuma-isaac-local/edge-refinement/window-research-56/RESULTS.md`:
  主窓5軸の回収率22.85%から91.83%、0.05 m診断89.85%。最終形状品質は不合格。
- 同`main-axis-07/combined/`: 元点から得た424本の形状初期値。
  `main-measured-07/`: 選別249本。`native-extra-axes/`: 支持監査後194本。
  最初は候補生成を凍結し、新旧の選択・再配置の効果だけ比較する。
- `src/usim_env_pipeline/window_member_sections.py`: 奥行き別の断面と原点ID。
- `src/usim_env_pipeline/window_axis_selection.py`: 既存競合と失敗した選別の比較。
  `--measured-sections`の現在経路は方向検査の数値を計算するが、適格条件では
  それを拒否条件に使わず全候補を通す。単なる閾値強化で以前の横枠欠落を再現しない。
- `window_member_sections.py`は断面ごとに近い成分を選び、原IDの和集合を返す。
  `window_member_axes.py`の`_refit`はその後、選択成分への所属を必須とせず
  軸の近くの元点を再収集する。支持の乗り換えを許す構造は確認したが、
  実際の誤線がどれだけこれに由来するかは未測定。断面列の所属を再構成して比較する。
- `src/usim_env_pipeline/simulation_facades.py`: 観測占有範囲で面を有限化する既存例。
  0.16 mの押出し厚は表示用仮定であり、今回の真の桟厚や物理稜線の正解には使わない。
- `src/usim_env_pipeline/simulation_primitive_detection.py`: 原ID付き面候補の既存例。
  0.1 m間引き・最小128点・最小面積0.5 m2を細い桟にそのまま適用しない。
- 過去の建物箱・FittedPlanes・補完済みsimulation形状を正解にはしない。
  元点から離れた壁や開口閉塞の問題があったため、既存成果物も比較専用に留める。

元XYZI・凍結LOF ID・v35物理線・v42部材軸を保護する。
浮島除去・地面再構成・衝突形状変更・全地図処理はこの計画の対象外。

## 一次研究の比較

| 研究 | 最適化するもの | 今回の使い方・限界 |
| --- | --- | --- |
| GlobFit, SIGGRAPH 2011 [R1] | 基本形状への局所適合と形状間関係 | 平行・共面などを点群適合と合わせる設計を参考にする。単一の直交系は強制しない |
| RAPTER, SIGGRAPH 2015 [R2] | 平面候補と関係の選択、データ説明と単純さ | 第一候補。小さな方向群を大きな壁方向へ潰さない。ただし原論文の評価は今回の窓軸精度の保証ではない |
| PolyFit, ICCV 2017 [R3] | 面候補の二値選択、適合・被覆・複雑さ | 比較候補。閉曲面と必要面が揃う前提は開いた窓cropと合わないため、そのまま主案にしない |
| Kinetic Shape Reconstruction / CGAL [R4] | 空間分割の内外ラベルから閉じた多角形面 | 壁外形の比較には有用。未観測部を面延長で埋めるので、桟の観測回収の証明にはしない |
| Screened Poisson, 2013 [R5] | 向き付き法線と点位置に合う陰関数 | メッシュ化基準線。平滑化や補完が細い桟・開口に与える影響を測る。主案にはしない |
| RFEPS, TOG 2022 [R12] | 局所平面性による除雑音、離散最適輸送による特徴線領域推定、特徴点追加と面再構成 | エッジを保つ形状化の重要な比較研究。ただし点の移動と追加は推定であり、元点の観測支持とは別に評価する |
| NeurCADRecon, SIGGRAPH 2024 [R13] | 点への適合とGaussian curvatureを0へ寄せる展開可能性の自己教師あり最適化 | 学習済み分類器以外の選択肢。自由曲面も必要な場合の第二候補。公式環境はCUDA/PyTorchで、今回のCPU主案とは分離する |
| Crease-Aware Normal Vectors, 2026-06-30 [R6] | 稜線を跨ぐ混合を避ける近傍部分集合 | 面の法線混合が支配的な場合だけ比較する。法線改善と窓部材回収を同一視しない |
| D-FRAME, TVCG 2025 [R7] | 学習したエッジ点と方向場からwireframe | CADの方向場設計は参考。ただし先にwireframe、後に面再構成という順序で、メッシュ先行の証拠ではない |
| Dual Contouring of Signed Distance Data, SIGGRAPH 2026 [R8] | 既にあるSDFから鋭い頂点を求める二次最適化 | SDF生成自体は解かない。今回のXYZIから符号・薄い桟のSDFが得られるとは仮定しない |
| Filmsticking++, 2026 preprint [R9] | 点を補間する明示的閉曲面 | 法線不要の比較研究。ただし雑音点も補間する目的と今回の形状推定は異なる。現データで未検証 |

これらの優劣は津田沼での実測順位ではない。上表の優先度は入力条件からの判断。
PolyFit公式コードはGPL-3.0、Gurobi/SCIP経路とWindows配布を確認したが、
ローカルでのビルド・起動・license適合検討は未実施。[R3-code]
主案では外部コードをコピーせず、既存CPU数値環境で最小の独立実験を作る。

## 形状化で何を持つか

### 表現

1. 壁や桟の観測表面を、原点支持範囲付きの有限平面パッチとして持つ。
2. 桟は細長い帯状パッチ、または観測された複数の断面面で表現する。
   初期軸は既存424候補から得るが、評価は軸チューブの内側点だけに限定しない。
3. 全点を何かの面へ押し込まず、クラッタ・外れ値・未説明のラベルを許す。
4. 前面しか見えない桟の裏面や閉じた箱を仮定しない。
   幅・中心・厚みが定まらないパラメータは未確定として残す。
5. ジャンクションは複数形状が接する領域として扱う。そこだけの方向混合で
   長い横材を丸ごと拒否しない一方、線全体を「交点だから不明」で通さない。

### 取り出す線のクラス

- 二つの独立に支持された面の有限交線: 物理稜線候補。
- 一枚の観測支持領域の有限境界: 観測境界。未観測側を実空間とは断定しない。
- 幅の両側が観測され、中心が安定している帯の中心線: 部材軸。
- 仮定だけで得た裏面・交点・閉鎖線: 推定形状。観測線の採点や回収率に混ぜない。

三角形の分割線、cropの切断端、押出しの裏側の辺は構造エッジとして出力しない。
ほぼ平行な二面の交線は位置が不安定になり得るため、元点による再fitの
安定性がないものは保留する。二面が観測されない場合、部材軸を物理稜線へ昇格しない。

## 最適化の設計仮説

目的は「正しい線を個別に採点する」から「点群を少数の整合する形状で説明する」への変更。
下式は今回提案する目的関数の構成であり、研究論文の式の引用ではない。

```text
min over geometry theta, selection z, assignments a, optional relations r:

  E_data(points -> observed finite patches, with outlier option)
  + lambda_support * E_support(observed patches -> source-supported extent)
  + lambda_boundary * E_boundary(observed transverse/depth profiles)
  + lambda_model * model_description_cost(z, theta)
  + lambda_relation * E_relation(theta, r)
```

- `E_data`: 同じ単位のHuber等の頑健残差。点から部材中心線までの距離ではなく、
  実際に観測された表面への距離を使う。点密度が高い壁が細い桟を圧倒しないよう、
  固定空間セル単位の重み上限を設ける。対応点と未説明点を保存する。
- `E_support`: 観測前面の支持範囲に対する外挿を罰する。未知の裏面に
  「点がないから空間である」というfree-space制約は置かない。
  スキャナ姿勢がないため視線ベースの空間消去は利用できない。
- `E_boundary`: 完全な横断面・奥行き断面で、帯の両側と位置が軸方向に
  持続するかを測る。単なる近接点数や接続数を加点する方式とは分ける。
  別の縦桟の一部だけを斜めに横切る候補に、同じ部材の持続として点を貸さない。
- `model_description_cost`: 形状数と自由度を罰し、同じ部材を多数の軸で
  重複説明する利益をなくす。長い線を一本出すだけで有利にはしない。
  パッチ外周の複雑さ・断片数・断面変動も記録し、無制限に穴だらけの一枚面へ
  全部材を吸収して形状数だけを減らす解を対照に含める。
- `E_relation`: 元点から独立に確認できる平行・共面などだけを任意の弱い制約にする。
  関係なしでも解ける基準線を必ず残す。直交・等間隔・矩形閉鎖は強制しない。

**最も重要な落とし穴:** データ距離と形状数だけでは、同一面の壁と桟を一枚に潰す解や、
幅の広い帯が他部材まで吸収する解が得られる。断面境界と複数奥行きの識別が
本当に増分情報を持つかを、最適化導入前に確認する。
元点が単一の平面帯しか示さない場合、物理厚みや真の中心を一意に回収できない。
その場合は推定値の安定範囲を示し、制約で無理に答えを作らない。

支持の会計は候補ごとでなく全体で行う。一つの原点・占有セルの説明利益を
複数候補へ重複加点しない。接合部では所属を保留または分配しても総重みは超えない。
同じ桟の別名を一群として除いたときに、他の形状で再最適化しても説明できない
観測がどれだけ残るかを診断する。偽斜線の点が既存の縦材・横材で説明済みなら
追加の形状を必要としない、という仮説を検証する。
個別候補に「独占点がないから削除」を適用して全別名を消す方式にはしない。
真の斜材でも独立の観測がなく同じ点群で二解が成立するならUNKNOWNとして残す。

### 解く順序

1. 候補形状と原点対応を固定して、離散的な形状選択を解く。
2. 選んだ観測形状の連続パラメータだけを頑健最小二乗で改善する。
3. 再対応・再選択を最大5回繰り返し、同じ目的値が改善しなければ終了。
4. 関係正則化なし/ありを比較する。端点延長は行わず、有限支持を再計算する。

既存SciPy 1.15.3のMILPと頑健least-squaresを実装候補とする。[R10][R11]
ペア関係が線形化できる離散部分だけをMILPで扱い、連続部分と合わせた
問題全体の大域最適性は主張しない。入力順を固定し、初期値を固定した3通りで比較する。
MILPは一成分120秒を初期予算とし、timeout、実行可能解、目的下界、gapを記録。
解けない場合は「最適解」と表示せず、不成立の条件も報告する。
観測上連結する長い部材を計算都合だけで切断しない。

## Todos

- [ ] 1. 評価領域と誤検知の測定契約を固定する。
  - Recommended task executor category: deep-low
  - 成果物: `runs/tsudanuma-isaac-local/edge-refinement/window-geometry-57/`
    の`INPUTS.json`, `REFERENCE_SCOPE.md`, `BASELINES.json`, `references/`。
  - 元の5軸と12軸は保持。既知southは以後回帰用であり、新しい盲検とは呼ばない。
  - 元点のみの複数視点・深度断面で、主窓と第二窓列の小さい採点体積を網羅注釈する。
    元点では確定できない区間はUNKNOWNとして先に固定。候補の有無で注釈を決めない。
    元参照の未支持上段は旧スコアを保持し、観測支持区間の補助スコアを別記する。
  - QA: 原ID・hash・領域・UNKNOWN一覧・クラスを照合。網羅性が確保できなければ、
    precisionはUNESTABLISHEDとして続け、採用合格は出さない。
  - Commit: 依頼されるまで行わない。実験データはruntimeに保持。

- [ ] 2. 形状識別が可能かを最小の局所比較で確認する。
  - Recommended task executor category: deep-high
  - 成果物: `identifiability.json`, `source-sections/`, `model-hypotheses/`。
  - 対象は元点から選ぶ横材・縦材・真の斜材対照・横切り偽線・壁接続・別奥行き。
  - 「一本の帯」「交差する複数の帯」「一枚の広い壁」を同じ局所点群へfitする。
    軸近傍点だけの誤差と完全断面の誤差を対比し、仮説が識別可能か確認する。
  - 元点を固定した0.30 m空間ブロック5分割で、fit用4分割と残り1分割を入れ替える。
    軸ブロック全欠測の実験も別に行い、近接点のランダム分割だけで安定と呼ばない。
    補助診断であり、窓全体の独立validationの代替にはしない。
  - QA: 正しい帯と横切り偽線を分けられず、幅・中心も不安定ならここで主案を止める。
    新しい候補を増やして失敗を隠さず、どの面・断面が観測不足か報告する。
  - Commit: 依頼されるまで行わない。

- [ ] 3. 同じ入力で三つの経路を比較する。
  - Recommended task executor category: deep-high
  - A: 凍結424候補の直接選別。旧249選別と194表示監査を区別した対照。
  - B: 有限平面パッチの選択・正則化と観測面の交線/境界抽出。
  - C: 有限面と帯状部材の混合最適化。今回の第一候補。
  - B/Cの内側には形状を固定した選択・所属最適化の対照を必ず置く。
    連続再配置がこの安価な対照を上回らなければ、複雑な再fitを採用しない。
  - B/Cは共通の元点・候補初期値・計算範囲で比較。原点へ独立にfitした面を使い、
    最終出力は同じ0.10 m支持監査へ通す。新旧どちらにも同じ表示フィルターを適用。
  - Cのablationは独立採点/共有会計、無限面/有限支持、固定形状/連続再fitを分離し、
    データのみ、境界/支持追加、複雑さ追加、弱い関係追加の順で比較する。
    目的項は単位と正規化を記録。重みの探索は各追加項0/0.1/1/10の座標探索に限定し、
    開発領域だけで選択後に設定hashを凍結する。
  - 同じ支持点の重複加点を防ぐ一方、実交点の複数部材と近接した独立面は保持する。
  - QA: `ablation.csv`, 項別目的値、原点対応、keep/drop理由、線分の親面/親部材IDを保存。
    旧成分IDを確定した部材所属とみなさず、断面列の所属と乗り換えを記録する。
    基準Aより正しい横材を落とすB/Cを採用しない。
  - Commit: 依頼されるまで行わない。

- [ ] 4. 同条件の局所メッシュ基準線で仮説を反証する。
  - Recommended task executor category: deep-low
  - B/Cが良く見えるだけでなく「メッシュ先行より有利か」を比較する。
    同じ小領域で、利用可能な既存メッシュ経路または隔離したPoisson対照を一つだけ使う。
    外部再構成コードの導入は環境・licenseを記録してから行い、本番依存へは追加しない。
  - 全三角形辺を出す方式にはしない。二面角のある稜線と観測境界を別抽出し、
    小さい三角形の分割辺、閉鎖用の仮想面、crop端を除く。
  - QA: 開口閉塞面積、桟の融合/消失、元点と面の双方向距離を保存。
    面誤差が小さくても窓線が消える場合は失敗とする。
    Poisson環境が利用できない場合は「比較未実行」とし、優越を主張しない。
  - Commit: 依頼されるまで行わない。

- [ ] 5. 設定凍結後の評価と実Isaac比較を行う。
  - Recommended task executor category: deep-low
  - 開発と原ID/構造が重ならない二つの新しい窓領域を元点だけで選定し、
    新方式を走らせる前に参照とUNKNOWNを凍結する。
    確保できなければ局所実験までとし、未見建物合格にはしない。
  - 既知main/west/south、植生、球、連続壁、実斜材、T/X接合、奥行きの異なる近接材、
    同一材の別名、壁接続材、観測ギャップを同じ設定で確認する。
  - QA: 下記の数値ゲート、全対応回帰、元点hashと独立再実行を確認。
    新規領域は開発で決めた一つの設定だけを評価し、結果を見て重みを戻さない。
    開発領域では採用重みの0.5倍/2倍でも回収非悪化と削減傾向が保たれるかを報告する。
    同じ視点で元点/A/B/C/形状のみ/線のみを撮影・目視し、確認した画像だけを数える。
    最終結果は実Isaac GUIで表示して開いたまま残す。既存GUIを勝手に閉じない。
  - Commit: 依頼されるまで行わない。

## 評価基準

全て提案する合否条件であり、達成済みの値ではない。

| 指標 | 条件 |
| --- | --- |
| 旧主窓5軸の0.10 m/5度回収率 | 91.83%から1 percentage point以内の低下、かつ90%以上 |
| 各旧参照軸 | Aに対し2 percentage pointsを超える低下を許さない |
| 0.05 m診断 | 全体と各軸を併記。0.10 mだけ良くなった位置ずれを検出 |
| 網羅注釈された局所体積の長さprecision/recall | P>=95%、R>=90%。UNKNOWNを無視した単一値だけで合格にしない |
| 明確な誤線長 | Aに対し80%以上削減を目標。単なる線本数削減ではない |
| 同一材の重複長 | Aに対し75%以上削減を目標。独立近接材は別扱い |
| 元点支持 | 従来と同じP95<=0.10 m、実ギャップ<=0.25 mを表示採用条件にする |
| 負例と既存v35/v42 | 新規偽線0、保護raw/座標/IDは不変 |
| 計算 | 時間、メモリ、最適化終了理由、gap、初期値間差を報告 |

長さprecisionはクラス別・有限区間別に計算する。一本の長い線の一部が一致しても、
残りをTPにしない。既知部材との多対一対応では最良対応分だけをTPとし、
残りの重複は別に計上する。独立した隣接材は同一物扱いしない。
UNKNOWNに重なる予測長と確定領域率を必ず報告する。
UNKNOWN予測を全て誤り/全て正しいと置く上下限も示し、都合の良い方だけを採用しない。
新規領域で網羅性が足りなければ「支持された候補の改善」と「精度証明」を分ける。

## 実装箇所と検査

実行承認後の候補:

- 数値部分は`src/usim_env_pipeline/window_geometry_fit.py`と
  `window_geometry_selection.py`へ分け、Isaac依存を入れない。
- 新しい数値テストは`test/test_env_window_geometry.py`に既存と同程度の粒度で置く。
- CPU実験CLIと監査は`runs/tsudanuma-isaac-local/`、
  各試行は`edge-refinement/window-geometry-57/`の新規名へ出力する。
- `window_contour_features.py`と候補生成コードは最初の比較では変えない。
  既存本番経路やsimulationの表示・衝突レイヤーへ接続しない。

テスト方針は数値回帰を先に失敗させてから実装する。
真の斜材、偽斜線、広い桟、T/X交点、壁接続、独立近接sheet、重複点、
点順変更、剛体変換、欠測、元点不変、全削除退化解を含める。
評価用GTやproseを固定するテストは追加しない。

検査コマンドは既存のPython 3.12 / pipeline extra / 単スレッド条件:

```text
uv run ruff check .
uv run ruff format --check .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra pipeline pytest -ra
uv build
```

実行はCPUを基本とし、GPU方式を採る別判断が必要になったらSlurmを使う。
Isaac表示の確認だけはローカルGPUで行う。commit/pushは依頼があるまで行わない。

## Final verification wave

- [ ] F1. 回収非悪化、誤線/重複削減、原点支持、クラス分離、全検査を照合する。
  - 証拠: `acceptance.json`, `metrics.csv`, 原ID監査、各checkの終了コード。
- [ ] F2. 新規領域と実Isaacの同一視点比較から採否を決める。
  - 証拠: `VALIDATION.md`, `render-final/manual-review.json`, 最終GUIの識別情報。
  - 成功なら局所採用範囲だけを明記。失敗なら最も良い非採用案と識別不能な構造を残す。

依存順: 1 -> 2 -> 3、4は2以降で独立比較、3/4 -> 5 -> F1/F2。

## 停止・方針変更

- 最小局所比較で形状仮説を識別できない: 大規模最適化へ進まず、観測不足を報告。
- 正則化で線は減るが横材も消える: 重みを強め続けず、その目的項を不採用にする。
- 良さそうな窓へ評価領域を移す、GTを後から修正する、southを再び未見扱いすることは禁止。
- B/CがAを上回らない: メッシュ化・最適化による改善は確認できなかった、と結論する。
- 推定された閉曲面を観測の代用にしない。完成CADの推定は別の成果物・別承認として扱う。
- 全地図展開や学習への拡張は局所採否を出した後の別計画。今回の計画完了は実装開始ではない。

## Sources

論文・著者ページ・公式実装/仕様のみ。検索要約だけで性能順位は付けていない。
下記URLは取得先であり、コードを導入・実行したことを意味しない。

- [R1] Li et al., GlobFit, SIGGRAPH 2011.
  `https://graphics.stanford.edu/~niloy/research/globFit/globFit_sigg11.html`
- [R2] Monszpart et al., RAPTER, SIGGRAPH 2015, 著者ページのabstract/algorithm/resultsを確認。
  `https://geometry.cs.ucl.ac.uk/projects/2015/regular-arrangements-of-planes/`
- [R3] Nan and Wonka, PolyFit, ICCV 2017, 著者ページのabstractを確認。
  PDF取得はInternal Errorとなり、本文全式を確認したとは扱わない。
  `https://3d.bk.tudelft.nl/liangliang/publications/2017/polyfit/polyfit.html`
- [R3-code] 著者の公式README: closed model / all necessary planes / solver / license.
  `https://github.com/LiangliangNan/PolyFit`
- [R4] CGAL著者解説、Bauchet and Lafarge, Kinetic Shape Reconstruction, TOG 2020.
  `https://cgal.github.io/cgal-web/2024/05/29/Kinetic_surface_reconstruction/`
- [R5] Kazhdan and Hoppe, Screened Poisson Surface Reconstruction, TOG 2013.
  著者ページのabstractを確認。本文PDFは未精読。
  `https://hhoppe.com/proj/screenedpoisson/`
- [R6] Law and Kang, Crease-Aware Normal Vectors from Unstructured Point Clouds, 2026.
  `https://link.springer.com/article/10.1007/s10851-026-01318-5`
- [R7] Feng et al., D-FRAME, TVCG 31(12), December 2025.
  `https://personal.ntu.edu.sg/yhe/papers/2025-tvcg-dframe.pdf`
- [R8] Carrera et al., Dual Contouring of Signed Distance Data, SIGGRAPH 2026.
  `https://gatc.cs.columbia.edu/projects/dual-contouring-of-signed-distance-data.html`
- [R9] Wang et al., Filmsticking++, arXiv preprint, 2026-02-11.
  `https://arxiv.org/abs/2602.11433`
- [R10] SciPy 1.15.3公式MILP仕様。
  `https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.milp.html`
- [R11] SciPy 1.15.3公式least_squares仕様。
  `https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html`
- [R12] Xu et al., RFEPS, TOG 2022, 著者のarXiv abstractと著者PDF検索結果を確認。
  `https://arxiv.org/abs/2212.03600`
- [R13] Dong et al., NeurCADRecon, SIGGRAPH 2024, 著者abstractと公式実装を確認。
  `https://arxiv.org/abs/2404.13420`
  `https://github.com/QiujieDong/NeurCADRecon`

## この計画作成時の確認範囲

一次資料・既存コード・失敗記録を照合し、独立したアルゴリズム相談を反映した。
別系統の設計相談はサービスの利用制限で実施できず、評価契約は主担当が点検した。
正式なplan-reviewer監査の合格を主張しない。
コード変更・点群最適化・新しいIsaac起動・GPU処理は行っていない。
文書のみの変更なのでテスト実行は追加していない。実測は実行後の成果物で確認する。
