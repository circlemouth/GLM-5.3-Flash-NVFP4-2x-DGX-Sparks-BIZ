# overlayの検証と受入

## 現在の状態

任意のoverlayに、元のBIZの本番受入結果は引き継がれません。
CPU試験は対象キーの選択、対象外のtupleとmetadataの保持、既定の無効状態、manifestと資産の検証、欠落と重複、非対応形状、派生checkpointの拒否を確認します。
構成を受け入れる前に、両rankでの全モデルロードとA/B/A′比較を別に記録します。

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
