# overlayの検証と受入

## 現在の状態

BF16 overlayは明示的に選ぶ機能で、通常BIZの受入結果を引き継ぎません。以下は固定したNVIDIA本体とdonorを使った2026-09-24のTP=2実機観測です。別環境の適格性や臨床利用の認証を示しません。接続先と未加工の実行記録は非公開で保持します。

## 2026-09-24の2台検証

ロードしたruntimeはfork commit `9a142366e5b128636eb22965a4192c1a6c1a43f7`と、両rankで同一のローカルimage ID `sha256:5c4f0b51a262ae519aa641634ccf63f775c9da47839735519555013eb56f86ab`を使いました。NVIDIA revisionは`423acf37583782c51c142d145aef733d72943d93`、Dealign revisionは`745aac2ff0f10acf961f396df3f9418598aa7327`、抽出資産manifestのSHA-256は`f6a9398f59a787ffee857ef89f86f71512362d1e9540fe560ad4fdd15342345a`です。後続の検証ツールだけのcommit `078c6df`は配備済みimageを変更しません。

| 確認項目 | 実測結果 | 限界 |
| --- | --- | --- |
| A：overlay無効 | 同じforkをTP=2で起動し、healthと短い合成API smokeに合格しました。 | 検証コマンドの実行パスを誤り、両rankのbase重み照合は完了していません。 |
| B：overlay有効 | TP=2、既存MTP深度3、262,144-token contextで起動しました。donor由来の本体29個とMTP L45は両rankで60/60一致し、対象外L44は両rankでNVIDIA baseと2/2一致しました。 | 格納値の検査対象は評価用launchで、別の常設launchではありません。 |
| A′：通常版への再切替 | 管理者がoverlay版の常設運用を選んだため実施していません。 | この一連の起動における通常版の再起動変動は未測定です。 |
| 合成院内タスク | C1 57/60、C2 57/60、critical 60/60、context 6/6でした。 | 保存済み通常BIZも合格件数は同じですが、harnessとworkloadのhashが異なり、同一Contract 5のA/B比較ではありません。 |
| 長文入力 | 実入力249,658 tokensから109 tokensを生成しました。 | 1件の合成probeであり、すべての長文入力の成功を保証しません。 |
| 思考モード | 認証付き直接APIのlow、high、maxで最終回答を各1回確認しました。 | 思考項目をすべて省略したraw HTTP要求1件はHTTP 200でも最終本文が空でした。BIZクライアントは別途、未指定をlowへ正規化します。 |

Bの評価と262K probeではOOM、CUDA crash、swap増加を認めませんでした。評価終了時、rank 1のvLLM workerはSIGTERM後のSIGKILLへ進みましたが、Dockerの`OOMKilled=false`、両GPUのcompute一覧は空、Controllerの終了gateは合格でした。その後、管理者の選択によりoverlay版を常設profileへ切り替えました。Controller revision 1717は受付open、leaseと警告なし、必須doctor全合格、両rank稼働を示しました。通常BIZのprofileと資産は切戻し用に保持しています。

検証ツールだけを修正した後のCPU試験は488件を実行し、442件合格、46件skipでした。Ruff checkと公開物検査は合格しました。repository全体のRuff formatは未変更の上流5ファイルを報告しますが、変更した検証ツール2ファイルのformat検査は合格です。品質、性能、未実施のA′は別に扱い、速度同等性や一般的な業務利用適格性を認定しません。

## 両rankの検証手順

ロード前に、稼働profile、所有者、受付状態、job、containerとimageのID、モデルrevision、メモリ、disk、fabric、ローカル改修、確認済みの復元手順を非公開の`records/`へ記録します。
結果を見る前に、試験基準と合成入力を固定します。
両ホストのGPUは一組として排他利用します。
制御された切替中は、業務trafficをdrainするか、受入済みの別経路へ退避します。

1. 確認済みfork commitから両ホストでimageをビルドし、本体とMTPのloader patchのsource hashと同じimage IDを確認します。
2. 両ホストでNVIDIA snapshot、MTP有効時のmetadata view、donor revision、manifest digest、抽出した各Safetensorsを検証します。
   各rankで`server preflight`を実行し、既存の派生checkpoint検査とGPU排他検査を維持します。
3. A（同じforkでoverlay無効）、B（同じimageで有効）、A′（再び無効）の順に、既存の切替手順で両rankを再起動します。
   TP=2、CPUとGPUの設定、context、KV、MTP深度、tokenizer、生成条件、warmup、cache条件をそろえ、cold load時間とTTFTを分けます。
4. 通常のTP分割を考慮して、両rankの格納済みparameterを調べます。
   本体29個とMTP有効時の第45層をdonor値と照合し、第44層とその他の対象外がbase値のままか確認します。
   MTP無効時に第45層を適用済みとは数えません。
   元ファイルの前後差を比較し、tensor本文はログに出しません。
5. 日本語回答、JSONとtool schema、mockの承認境界、tool result、SSE、停止理由、多ターン、応答異常を合成ケースで試します。
   thinking offの速度とhigh/maxの機能は分けて調べ、基準を変えずに成否を記録します。
   固定長に近い速度測定とworkflow完了時間も分けます。
6. APIとtaskの成功率、TTFT、decode速度、workflowのp50/p95、token数、MTP採択、両ホストのロード時と稼働時のメモリ余裕、GPUと通信のエラーを測ります。
   主指標が10%以上悪化した場合は限定した再測定で確認します。
   短文が通ってから合成入力を32K、64K、開始時の実運用目標長へ段階的に伸ばし、対応上限を超えません。
7. 開始時のサービス構成を復元し、API healthと合成smoke requestを確認します。
   試験用processが残っていないことも確認します。

overlayの正しさ、通常版の非回帰、観察したモデル挙動、task品質、速度、復元を別々に報告します。
再起動をまたぐ応答差だけではoverlayの効果を示せません。
この手順は業務、臨床、法務の適合を認証しません。
