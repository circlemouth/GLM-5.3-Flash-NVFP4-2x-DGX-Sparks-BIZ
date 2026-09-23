# overlayの出所とライセンス

このforkは、BIZのApache-2.0の`LICENSE`、`NOTICE`、`LICENSES/`、第三者通知を保持します。
新しいoverlayコードにはApache-2.0のSPDX表示を付けています。
改変対象のvLLM loaderは元のApache-2.0表示を残し、このforkによる変更を明示します。
既存のMITとApacheの表示は、各構成要素に引き続き適用されます。

重みはコードとは別に運用者が取得する資産であり、repository、image、releaseには含めません。
固定revisionの[NVIDIAモデルカード](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/423acf37583782c51c142d145aef733d72943d93/README.md)はNVFP4 checkpointをMITと表示し、商用と非商用の利用に言及しています。
同snapshotには独立したLICENSEファイルがありません。
NVIDIAが親モデルとして示す[Z.AIのGLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)にはMITのLICENSEがあります。
重みを再配布する場合は、該当する著作権表示、許諾文、免責文を保持する必要があります。

任意のdonorは[Dealignの固定revision](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4/tree/745aac2ff0f10acf961f396df3f9418598aa7327)です。
その[LICENSE](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4/blob/745aac2ff0f10acf961f396df3f9418598aa7327/LICENSE)はMITで、Z.AIの著作権表示を含みます。
モデルカードもMITと表示しています。
一方、編集された各テンソルの由来を最後まで追える資料はモデルカードにありません。
このforkは、すべての上流の権利と再配布時の表示条件が解決済みとは認定しません。
ローカルの抽出manifestにdonorのcommitとテンソルのハッシュを記録します。

方式の参考となったMiaAI-Labの公開レシピは[READMEの比較表](../README.ja.md#dgx-spark向けの他のglm-53-flashレシピ)に示します。
Miaのコード、script、EXL3 runtime、モデルファイル、test、文書本文は取り込んでいません。
この機能にAGPL由来のコードは追加していません。
AGPL自体を商用利用禁止とは扱いません。

overlayはソフトウェアとモデルの実験であり、法務、医療、安全、臨床の認証ではありません。
運用者はソースのライセンス、モデルの条件、利用する環境に応じた義務を確認してください。
ビルド後のコンテナ内の依存物には、それぞれの条件が適用されます。
