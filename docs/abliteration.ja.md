# BF16 o_proj重みの任意overlay

この非公式BIZ forkは、NVIDIAの固定NVFP4 checkpointをロードするとき、本体の第15～43層にある`self_attn.o_proj.weight`の29個と、標準MTPの第45層にある1個を置換します。
第0～14層、第44層、その他のattention重み、experts、vision、tokenizer、chat template、API契約は変更しません。
`runtime.weight_overlay.enabled = true`を指定しない限り無効です。
重みとコンテナ画像は、このforkでは配布しません。

donorの固定revisionは[資産manifest](../config/abliteration.lock.json)に記録します。
出所と条件は[ライセンス文書](abliteration-licensing.ja.md)、検証状況は[検証文書](abliteration-validation.ja.md)に記載します。
元のBIZに対する受入結果を、この任意構成の受入結果として扱うことはできません。

## 資産の準備

`config/runtime.lock.json`と`config/abliteration.lock.json`の固定revisionを使います。
抽出コマンドは両モデルのindexとSafetensorsヘッダーを読み、対象のBF16データだけを取得します。
HTTP部分応答を検証し、排他ロックの下で1テンソルずつSafetensorsへ保存します。
生成した`manifest.json`と重みファイルはGitの外に置きます。

```sh
python tools/prepare_abliteration.py --help
python tools/prepare_abliteration.py inspect --output /private/overlay-assets
python tools/prepare_abliteration.py extract --output /private/overlay-assets
sha256sum /private/overlay-assets/manifest.json
```

両ホストで同じ抽出を行うか、再開と検証ができる方法で転送し、各ホストの30ファイルのハッシュを照合します。
donorの`config.json`、tokenizer、chat templateはNVIDIAのsnapshotへ混ぜません。
MTPを使う場合は、既存の`tools/prepare_mtp_view.py`で読み取り専用のmetadata viewを準備します。

## 構成の選択

通常の`examples/server.example.toml`にはoverlayの設定表がなく、donorファイルを開きません。
overlay構成では既存設定を保ち、`[runtime]`の下に次の表を追加します。
パスとdigestは実機で検証した値に置き換え、`api.served_model_name`には通常版と異なる名前を指定します。
両ホストには、このforkの確認済みcommitから作成した同じimage IDを設定します。

```toml
[runtime.weight_overlay]
enabled = true
path = "/private/overlay-assets"
donor_revision = "745aac2ff0f10acf961f396df3f9418598aa7327"
manifest_sha256 = "<小文字の16進数64文字>"
```

ランチャーは資産を`/weight-overlay`へ読み取り専用でマウントします。
構成のfingerprintにはoverlay設定、donor revision、manifest digest、NVIDIAの固定revisionを含めます。
起動時の識別情報には、確認したimage IDも記録します。
preflightは資産を検証し、対象外のAXL派生checkpointを拒否します。
両rankは既存の[復元可能な切替手順](launch-safety.ja.md#全rail検査と2rank切替)で同時に切り替えます。
この切替で両rankが再起動するため、KVとprefix cacheの状態は引き継ぎません。
MTPを無効にした場合、第45層はロードされず、適用数は29個です。

imageのsource hash検査は、想定外のvLLM本体またはMTP loaderへのpatch適用を拒否します。
ロード中に対象が欠ける、重複する、変更される、BF16以外になる、形状が異なる場合は起動を失敗させます。
元のcheckpointは編集しません。
起動成功だけでは両rankに格納された値を証明できないため、採用前に[検証手順](abliteration-validation.ja.md)を実行します。
