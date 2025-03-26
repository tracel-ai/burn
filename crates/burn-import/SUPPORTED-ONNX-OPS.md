# Supported ONNX Operators

Note: some ONNX Ops listed below are pseudo Ops, such as `Linear`, `Conv1d`, `Conv2d` (or other with
1d, 2d suffixes used signify the dimensionality). These are not real ONNX Ops, but are used to
represent the corresponding Burn Op.

| ONNX OP                          | Import Support | Burn Support |
|----------------------------------|:--------------:|:------------:|
| [Abs][1]                         | ✅             | ✅           |
| [Acos][2]                        | ❌             | ❌           |
| [Acosh][3]                       | ❌             | ❌           |
| [Add][4]                         | ✅             | ✅           |
| [And][5]                         | ❌             | ❌           |
| [ArgMax][6]                      | ✅             | ✅           |
| [ArgMin][7]                      | ❌             | ❌           |
| [Asin][8]                        | ❌             | ❌           |
| [Asinh][9]                       | ❌             | ❌           |
| [Atan][10]                       | ❌             | ❌           |
| [Atanh][11]                      | ❌             | ❌           |
| [AveragePool1d][12]              | ✅             | ✅           |
| [AveragePool2d][12]              | ✅             | ✅           |
| [BatchNormalization][14]         | ✅             | ✅           |
| [Bernoulli][15]                  | ❌             | ❌           |
| [BitShift][16]                   | ❌             | ❌           |
| [BitwiseAnd][17]                 | ❌             | ❌           |
| [BitwiseNot][18]                 | ❌             | ❌           |
| [BitwiseOr][19]                  | ❌             | ❌           |
| [BitwiseXor][20]                 | ❌             | ❌           |
| [BlackmanWindow][21]             | ❌             | ❌           |
| [Cast][22]                       | ✅             | ✅           |
| [CastLike][23]                   | ❌             | ❌           |
| [Ceil][24]                       | ❌             | ❌           |
| [Celu][25]                       | ❌             | ❌           |
| [CenterCropPad][26]              | ❌             | ❌           |
| [Clip][27]                       | ✅             | ✅           |
| [Col2Im][28]                     | ❌             | ❌           |
| [Compress][29]                   | ❌             | ❌           |
| [Concat][30]                     | ✅             | ✅           |
| [ConcatFromSequence][31]         | ❌             | ❌           |
| [Constant][32]                   | ✅             | ✅           |
| [ConstantOfShape][33]            | ✅             | ✅           |
| [Conv1d][34]                     | ✅             | ✅           |
| [Conv2d][34]                     | ✅             | ✅           |
| [Conv3d][34]                     | ✅             | ✅           |
| [ConvInteger][37]                | ❌             | ❌           |
| [ConvTranspose1d][38]            | ✅             | ✅           |
| [ConvTranspose2d][38]            | ✅             | ✅           |
| [ConvTranspose3d][38]            | ✅             | ✅           |
| [Cos][39]                        | ✅             | ✅           |
| [Cosh][40]                       | ✅             | ✅           |
| [CumSum][41]                     | ❌             | ❌           |
| [DepthToSpace][42]               | ❌             | ❌           |
| [DequantizeLinear][43]           | ❌             | ❌           |
| [Det][44]                        | ❌             | ❌           |
| [DFT][45]                        | ❌             | ❌           |
| [Div][46]                        | ✅             | ✅           |
| [Dropout][47]                    | ✅             | ✅           |
| [DynamicQuantizeLinear][48]      | ❌             | ❌           |
| [Einsum][49]                     | ❌             | ❌           |
| [Elu][50]                        | ❌             | ❌           |
| [Equal][51]                      | ✅             | ✅           |
| [Erf][52]                        | ✅             | ✅           |
| [Exp][53]                        | ✅             | ✅           |
| [Expand][54]                     | ✅             | ✅           |
| [EyeLike][55]                    | ❌             | ❌           |
| [Flatten][56]                    | ✅             | ✅           |
| [Floor][57]                      | ✅             | ✅           |
| [Gather][58]                     | ✅             | ✅           |
| [GatherElements][59]             | ✅             | ✅           |
| [GatherND][60]                   | ❌             | ❌           |
| [Gelu][61]                       | ✅             | ✅           |
| [Gemm][62]                       | ✅             | 🟨           |
| [GlobalAveragePool][63]          | ✅             | ✅           |
| [GlobalLpPool][64]               | ❌             | ❌           |
| [GlobalMaxPool][65]              | ❌             | ❌           |
| [Greater][66]                    | ✅             | ✅           |
| [GreaterOrEqual][67]             | ✅             | ✅           |
| [GridSample][68]                 | ❌             | ❌           |
| [GroupNormalization][69]         | ❌             | ✅           |
| [GRU][70]                        | ❌             | ✅           |
| [HammingWindow][71]              | ❌             | ❌           |
| [HannWindow][72]                 | ❌             | ❌           |
| [Hardmax][73]                    | ❌             | ❌           |
| [HardSigmoid][74]                | ✅             | ✅           |
| [HardSwish][75]                  | ❌             | ❌           |
| [Identity][76]                   | ✅             | ✅           |
| [If][77]                         | ❌             | ✅           |
| [Im][78]                         | ❌             | ❌           |
| [InstanceNormalization][79]      | ❌             | ✅           |
| [IsInf][80]                      | ❌             | ❌           |
| [IsNaN][81]                      | ❌             | ❌           |
| [LayerNormalization][82]         | ✅             | ✅           |
| [LeakyRelu][83]                  | ✅             | ✅           |
| [Less][84]                       | ✅             | ✅           |
| [LessOrEqual][85]                | ✅             | ✅           |
| Linear                           | ✅             | ✅           |
| [Log][87]                        | ✅             | ✅           |
| [LogSoftmax][88]                 | ✅             | ✅           |
| [Loop][89]                       | ❌             | ❌           |
| [LpNormalization][90]            | ❌             | ❌           |
| [LpPool][91]                     | ❌             | ❌           |
| [LRN][92]                        | ❌             | ❌           |
| [LSTM][93]                       | ❌             | ✅           |
| [MatMul][94]                     | ✅             | ✅           |
| [MatMulInteger][95]              | ❌             | ✅           |
| [Max][96]                        | ✅             | ✅           |
| [MaxPool1d][97]                  | ✅             | ✅           |
| [MaxPool2d][98]                  | ✅             | ✅           |
| [MaxRoiPool][99]                 | ❌             | ❌           |
| [MaxUnpool][100]                 | ❌             | ❌           |
| [Mean][101]                      | ✅             | ✅           |
| [MeanVarianceNormalization][102] | ❌             | ❌           |
| [MelWeightMatrix][103]           | ❌             | ❌           |
| [Min][104]                       | ✅             | ✅           |
| [Mish][105]                      | ❌             | ❌           |
| [Mod][106]                       | ❌             | ❌           |
| [Mul][107]                       | ✅             | ✅           |
| [Multinomial][108]               | ❌             | ❌           |
| [Neg][109]                       | ✅             | ✅           |
| [NegativeLogLikelihoodLoss][110] | ❌             | ❌           |
| [NonMaxSuppression][112]         | ❌             | ❌           |
| [NonZero][113]                   | ❌             | ❌           |
| [Not][114]                       | ✅             | ✅           |
| [OneHot][115]                    | ✅             | ✅           |
| [Optional][116]                  | ❌             | ❌           |
| [OptionalGetElement][117]        | ❌             | ❌           |
| [OptionalHasElement][118]        | ❌             | ❌           |
| [Or][119]                        | ❌             | ❌           |
| [Pad][120]                       | ✅             | ✅           |
| [Pow][121]                       | ✅             | ✅           |
| [PRelu][122]                     | ✅             | ✅           |
| [QLinearConv][123]               | ❌             | ❌           |
| [QLinearMatMul][124]             | ❌             | ❌           |
| [QuantizeLinear][125]            | ❌             | ❌           |
| [RandomNormal][126]              | ✅             | ✅           |
| [RandomNormalLike][127]          | ✅             | ✅           |
| [RandomUniform][128]             | ✅             | ✅           |
| [RandomUniformLike][129]         | ✅             | ✅           |
| [Range][130]                     | ✅             | ✅           |
| [Reciprocal][131]                | ✅             | ✅           |
| [ReduceL][132]                   | ❌             | ❌           |
| [ReduceLogSum][133]              | ❌             | ❌           |
| [ReduceLogSumExp][134]           | ❌             | ❌           |
| [ReduceMax][135]                 | ✅             | ✅           |
| [ReduceMean][136]                | ✅             | ✅           |
| [ReduceMin][137]                 | ✅             | ✅           |
| [ReduceProd][138]                | ✅             | ✅           |
| [ReduceSum][139]                 | ✅             | ✅           |
| [ReduceSumSquare][140]           | ❌             | ❌           |
| [Relu][141]                      | ✅             | ✅           |
| [Reshape][142]                   | ✅             | ✅           |
| [Resize][143]                    | ✅             | ✅           |
| [ReverseSequence][144]           | ❌             | ❌           |
| [RNN][145]                       | ❌             | ✅           |
| [RoiAlign][146]                  | ❌             | ❌           |
| [Round][147]                     | ❌             | ❌           |
| [Scan][148]                      | ❌             | ❌           |
| [Scatter][149]                   | ❌             | ✅           |
| [ScatterElements][150]           | ❌             | ❌           |
| [ScatterND][151]                 | ❌             | ❌           |
| [Selu][152]                      | ❌             | ❌           |
| [SequenceAt][153]                | ❌             | ❌           |
| [SequenceConstruct][154]         | ❌             | ❌           |
| [SequenceEmpty][155]             | ❌             | ❌           |
| [SequenceErase][156]             | ❌             | ❌           |
| [SequenceInsert][157]            | ❌             | ❌           |
| [SequenceLength][158]            | ❌             | ❌           |
| [SequenceMap][159]               | ❌             | ❌           |
| [Shape][160]                     | ✅             | ✅           |
| [Shrink][161]                    | ❌             | ❌           |
| [Sigmoid][162]                   | ✅             | ✅           |
| [Sign][163]                      | ✅             | ✅           |
| [Sin][164]                       | ✅             | ✅           |
| [Sinh][165]                      | ✅             | ✅           |
| [Size][166]                      | ❌             | ❌           |
| [Slice][167]                     | ✅             | ✅           |
| [Softmax][168]                   | ✅             | ✅           |
| [SoftmaxCrossEntropyLoss][169]   | ❌             | ❌           |
| [Softplus][170]                  | ❌             | ❌           |
| [Softsign][171]                  | ❌             | ❌           |
| [SpaceToDepth][172]              | ❌             | ❌           |
| [Split][173]                     | ✅             | ✅           |
| [SplitToSequence][174]           | ❌             | ❌           |
| [Sqrt][175]                      | ✅             | ✅           |
| [Squeeze][176]                   | ✅             | ✅           |
| [STFT][177]                      | ❌             | ❌           |
| [StringNormalizer][178]          | ❌             | ❌           |
| [Sub][179]                       | ✅             | ✅           |
| [Sum][180]                       | ✅             | ✅           |
| [Tan][181]                       | ✅             | ✅           |
| [Tanh][182]                      | ✅             | ✅           |
| [TfIdfVectorizer][183]           | ❌             | ❌           |
| [ThresholdedRelu][184]           | ❌             | ❌           |
| [Tile][185]                      | ✅             | ✅           |
| [TopK][186]                      | ✅             | ✅           |
| [Transpose][187]                 | ✅             | ✅           |
| [Trilu][188]                     | ✅             | ✅           |
| [Unique][189]                    | ❌             | ❌           |
| [Upsample][190]                  | ❌             | ❌           |
| [Where][191]                     | ✅             | ✅           |
| [Xor][192]                       | ❌             | ❌           |
| [Unsqueeze][193]                 | ✅             | ✅           |

🟨 Supported via other operations.

[1]: https://onnx.ai/onnx/operators/onnx__Abs.html "ONNX Abs"
[2]: https://onnx.ai/onnx/operators/onnx__Acos.html "ONNX Acos"
[3]: https://onnx.ai/onnx/operators/onnx__Acosh.html "ONNX Acosh"
[4]: https://onnx.ai/onnx/operators/onnx__Add.html "ONNX Add"
[5]: https://onnx.ai/onnx/operators/onnx__And.html "ONNX And"
[6]: https://onnx.ai/onnx/operators/onnx__ArgMax.html "ONNX ArgMax"
[7]: https://onnx.ai/onnx/operators/onnx__ArgMin.html "ONNX ArgMin"
[8]: https://onnx.ai/onnx/operators/onnx__Asin.html "ONNX Asin"
[9]: https://onnx.ai/onnx/operators/onnx__Asinh.html "ONNX Asinh"
[10]: https://onnx.ai/onnx/operators/onnx__Atan.html "ONNX Atan"
[11]: https://onnx.ai/onnx/operators/onnx__Atanh.html "ONNX Atanh"
[12]: https://onnx.ai/onnx/operators/onnx__AveragePool.html "ONNX AveragePool"
[14]: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html "ONNX BatchNormalization"
[15]: https://onnx.ai/onnx/operators/onnx__Bernoulli.html "ONNX Bernoulli"
[16]: https://onnx.ai/onnx/operators/onnx__BitShift.html "ONNX BitShift"
[17]: https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html "ONNX BitwiseAnd"
[18]: https://onnx.ai/onnx/operators/onnx__BitwiseNot.html "ONNX BitwiseNot"
[19]: https://onnx.ai/onnx/operators/onnx__BitwiseOr.html "ONNX BitwiseOr"
[20]: https://onnx.ai/onnx/operators/onnx__BitwiseXor.html "ONNX BitwiseXor"
[21]: https://onnx.ai/onnx/operators/onnx__BlackmanWindow.html "ONNX BlackmanWindow"
[22]: https://onnx.ai/onnx/operators/onnx__Cast.html "ONNX Cast"
[23]: https://onnx.ai/onnx/operators/onnx__CastLike.html "ONNX CastLike"
[24]: https://onnx.ai/onnx/operators/onnx__Ceil.html "ONNX Ceil"
[25]: https://onnx.ai/onnx/operators/onnx__Celu.html "ONNX Celu"
[26]: https://onnx.ai/onnx/operators/onnx__CenterCropPad.html "ONNX CenterCropPad"
[27]: https://onnx.ai/onnx/operators/onnx__Clip.html "ONNX Clip"
[28]: https://onnx.ai/onnx/operators/onnx__Col2Im.html "ONNX Col2Im"
[29]: https://onnx.ai/onnx/operators/onnx__Compress.html "ONNX Compress"
[30]: https://onnx.ai/onnx/operators/onnx__Concat.html "ONNX Concat"
[31]: https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html "ONNX ConcatFromSequence"
[32]: https://onnx.ai/onnx/operators/onnx__Constant.html "ONNX Constant"
[33]: https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html "ONNX ConstantOfShape"
[34]: https://onnx.ai/onnx/operators/onnx__Conv.html "ONNX Conv"
[37]: https://onnx.ai/onnx/operators/onnx__ConvInteger.html "ONNX ConvInteger"
[38]: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html "ONNX ConvTranspose"
[39]: https://onnx.ai/onnx/operators/onnx__Cos.html "ONNX Cos"
[40]: https://onnx.ai/onnx/operators/onnx__Cosh.html "ONNX Cosh"
[41]: https://onnx.ai/onnx/operators/onnx__CumSum.html "ONNX CumSum"
[42]: https://onnx.ai/onnx/operators/onnx__DepthToSpace.html "ONNX DepthToSpace"
[43]: https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html "ONNX DequantizeLinear"
[44]: https://onnx.ai/onnx/operators/onnx__Det.html "ONNX Det"
[45]: https://onnx.ai/onnx/operators/onnx__DFT.html "ONNX DFT"
[46]: https://onnx.ai/onnx/operators/onnx__Div.html "ONNX Div"
[47]: https://onnx.ai/onnx/operators/onnx__Dropout.html "ONNX Dropout"
[48]: https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html "ONNX DynamicQuantizeLinear"
[49]: https://onnx.ai/onnx/operators/onnx__Einsum.html "ONNX Einsum"
[50]: https://onnx.ai/onnx/operators/onnx__Elu.html "ONNX Elu"
[51]: https://onnx.ai/onnx/operators/onnx__Equal.html "ONNX Equal"
[52]: https://onnx.ai/onnx/operators/onnx__Erf.html "ONNX Erf"
[53]: https://onnx.ai/onnx/operators/onnx__Exp.html "ONNX Exp"
[54]: https://onnx.ai/onnx/operators/onnx__Expand.html "ONNX Expand"
[55]: https://onnx.ai/onnx/operators/onnx__EyeLike.html "ONNX EyeLike"
[56]: https://onnx.ai/onnx/operators/onnx__Flatten.html "ONNX Flatten"
[57]: https://onnx.ai/onnx/operators/onnx__Floor.html "ONNX Floor"
[58]: https://onnx.ai/onnx/operators/onnx__Gather.html "ONNX Gather"
[59]: https://onnx.ai/onnx/operators/onnx__GatherElements.html "ONNX GatherElements"
[60]: https://onnx.ai/onnx/operators/onnx__GatherND.html "ONNX GatherND"
[61]: https://onnx.ai/onnx/operators/onnx__Gelu.html "ONNX Gelu"
[62]: https://onnx.ai/onnx/operators/onnx__Gemm.html "ONNX Gemm (Linear Layer)"
[63]: https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html "ONNX GlobalAveragePool"
[64]: https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html "ONNX GlobalLpPool"
[65]: https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html "ONNX GlobalMaxPool"
[66]: https://onnx.ai/onnx/operators/onnx__Greater.html "ONNX Greater"
[67]: https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html "ONNX GreaterOrEqual"
[68]: https://onnx.ai/onnx/operators/onnx__GridSample.html "ONNX GridSample"
[69]: https://onnx.ai/onnx/operators/onnx__GroupNormalization.html "ONNX GroupNormalization"
[70]: https://onnx.ai/onnx/operators/onnx__GRU.html "ONNX GRU"
[71]: https://onnx.ai/onnx/operators/onnx__HammingWindow.html "ONNX HammingWindow"
[72]: https://onnx.ai/onnx/operators/onnx__HannWindow.html "ONNX HannWindow"
[73]: https://onnx.ai/onnx/operators/onnx__Hardmax.html "ONNX Hardmax"
[74]: https://onnx.ai/onnx/operators/onnx__HardSigmoid.html "ONNX HardSigmoid"
[75]: https://onnx.ai/onnx/operators/onnx__HardSwish.html "ONNX HardSwish"
[76]: https://onnx.ai/onnx/operators/onnx__Identity.html "ONNX Identity"
[77]: https://onnx.ai/onnx/operators/onnx__If.html "ONNX If"
[78]: https://onnx.ai/onnx/operators/onnx__Im.html "ONNX Im"
[79]: https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html "ONNX InstanceNormalization"
[80]: https://onnx.ai/onnx/operators/onnx__IsInf.html "ONNX IsInf"
[81]: https://onnx.ai/onnx/operators/onnx__IsNaN.html "ONNX IsNaN"
[82]: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html "ONNX LayerNormalization"
[83]: https://onnx.ai/onnx/operators/onnx__LeakyRelu.html "ONNX LeakyRelu"
[84]: https://onnx.ai/onnx/operators/onnx__Less.html "ONNX Less"
[85]: https://onnx.ai/onnx/operators/onnx__LessOrEqual.html "ONNX LessOrEqual"
[87]: https://onnx.ai/onnx/operators/onnx__Log.html "ONNX Log"
[88]: https://onnx.ai/onnx/operators/onnx__LogSoftmax.html "ONNX LogSoftmax"
[89]: https://onnx.ai/onnx/operators/onnx__Loop.html "ONNX Loop"
[90]: https://onnx.ai/onnx/operators/onnx__LpNormalization.html "ONNX LpNormalization"
[91]: https://onnx.ai/onnx/operators/onnx__LpPool.html "ONNX LpPool"
[92]: https://onnx.ai/onnx/operators/onnx__LRN.html "ONNX LRN"
[93]: https://onnx.ai/onnx/operators/onnx__LSTM.html "ONNX LSTM"
[94]: https://onnx.ai/onnx/operators/onnx__MatMul.html "ONNX MatMul"
[95]: https://onnx.ai/onnx/operators/onnx__MatMulInteger.html "ONNX MatMulInteger"
[96]: https://onnx.ai/onnx/operators/onnx__Max.html "ONNX Max"
[97]: https://onnx.ai/onnx/operators/onnx__MaxPool1d.html "ONNX MaxPool1d"
[98]: https://onnx.ai/onnx/operators/onnx__MaxPool2d.html "ONNX MaxPool2d"
[99]: https://onnx.ai/onnx/operators/onnx__MaxRoiPool.html "ONNX MaxRoiPool"
[100]: https://onnx.ai/onnx/operators/onnx__MaxUnpool.html "ONNX MaxUnpool"
[101]: https://onnx.ai/onnx/operators/onnx__Mean.html "ONNX Mean"
[102]: https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html "ONNX MeanVarianceNormalization"
[103]: https://onnx.ai/onnx/operators/onnx__MelWeightMatrix.html "ONNX MelWeightMatrix"
[104]: https://onnx.ai/onnx/operators/onnx__Min.html "ONNX Min"
[105]: https://onnx.ai/onnx/operators/onnx__Mish.html "ONNX Mish"
[106]: https://onnx.ai/onnx/operators/onnx__Mod.html "ONNX Mod"
[107]: https://onnx.ai/onnx/operators/onnx__Mul.html "ONNX Mul"
[108]: https://onnx.ai/onnx/operators/onnx__Multinomial.html "ONNX Multinomial"
[109]: https://onnx.ai/onnx/operators/onnx__Neg.html "ONNX Neg"
[110]: https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html "ONNX NegativeLogLikelihoodLoss"
[112]: https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html "ONNX NonMaxSuppression"
[113]: https://onnx.ai/onnx/operators/onnx__NonZero.html "ONNX NonZero"
[114]: https://onnx.ai/onnx/operators/onnx__Not.html "ONNX Not"
[115]: https://onnx.ai/onnx/operators/onnx__OneHot.html "ONNX OneHot"
[116]: https://onnx.ai/onnx/operators/onnx__Optional.html "ONNX Optional"
[117]: https://onnx.ai/onnx/operators/onnx__OptionalGetElement.html "ONNX OptionalGetElement"
[118]: https://onnx.ai/onnx/operators/onnx__OptionalHasElement.html "ONNX OptionalHasElement"
[119]: https://onnx.ai/onnx/operators/onnx__Or.html "ONNX Or"
[120]: https://onnx.ai/onnx/operators/onnx__Pad.html "ONNX Pad"
[121]: https://onnx.ai/onnx/operators/onnx__Pow.html "ONNX Pow"
[122]: https://onnx.ai/onnx/operators/onnx__PRelu.html "ONNX PRelu"
[123]: https://onnx.ai/onnx/operators/onnx__QLinearConv.html "ONNX QLinearConv"
[124]: https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html "ONNX QLinearMatMul"
[125]: https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html "ONNX QuantizeLinear"
[126]: https://onnx.ai/onnx/operators/onnx__RandomNormal.html "ONNX RandomNormal"
[127]: https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html "ONNX RandomNormalLike"
[128]: https://onnx.ai/onnx/operators/onnx__RandomUniform.html "ONNX RandomUniform"
[129]: https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html "ONNX RandomUniformLike"
[130]: https://onnx.ai/onnx/operators/onnx__Range.html "ONNX Range"
[131]: https://onnx.ai/onnx/operators/onnx__Reciprocal.html "ONNX Reciprocal"
[132]: https://onnx.ai/onnx/operators/onnx__ReduceL.html "ONNX ReduceL"
[133]: https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html "ONNX ReduceLogSum"
[134]: https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html "ONNX ReduceLogSumExp"
[135]: https://onnx.ai/onnx/operators/onnx__ReduceMax.html "ONNX ReduceMax"
[136]: https://onnx.ai/onnx/operators/onnx__ReduceMean.html "ONNX ReduceMean"
[137]: https://onnx.ai/onnx/operators/onnx__ReduceMin.html "ONNX ReduceMin"
[138]: https://onnx.ai/onnx/operators/onnx__ReduceProd.html "ONNX ReduceProd"
[139]: https://onnx.ai/onnx/operators/onnx__ReduceSum.html "ONNX ReduceSum"
[140]: https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html "ONNX ReduceSumSquare"
[141]: https://onnx.ai/onnx/operators/onnx__Relu.html "ONNX Relu"
[142]: https://onnx.ai/onnx/operators/onnx__Reshape.html "ONNX Reshape"
[143]: https://onnx.ai/onnx/operators/onnx__Resize.html "ONNX Resize"
[144]: https://onnx.ai/onnx/operators/onnx__ReverseSequence.html "ONNX ReverseSequence"
[145]: https://onnx.ai/onnx/operators/onnx__RNN.html "ONNX RNN"
[146]: https://onnx.ai/onnx/operators/onnx__RoiAlign.html "ONNX RoiAlign"
[147]: https://onnx.ai/onnx/operators/onnx__Round.html "ONNX Round"
[148]: https://onnx.ai/onnx/operators/onnx__Scan.html "ONNX Scan"
[149]: https://onnx.ai/onnx/operators/onnx__Scatter.html "ONNX Scatter"
[150]: https://onnx.ai/onnx/operators/onnx__ScatterElements.html "ONNX ScatterElements"
[151]: https://onnx.ai/onnx/operators/onnx__ScatterND.html "ONNX ScatterND"
[152]: https://onnx.ai/onnx/operators/onnx__Selu.html "ONNX Selu"
[153]: https://onnx.ai/onnx/operators/onnx__SequenceAt.html "ONNX SequenceAt"
[154]: https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html "ONNX SequenceConstruct"
[155]: https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html "ONNX SequenceEmpty"
[156]: https://onnx.ai/onnx/operators/onnx__SequenceErase.html "ONNX SequenceErase"
[157]: https://onnx.ai/onnx/operators/onnx__SequenceInsert.html "ONNX SequenceInsert"
[158]: https://onnx.ai/onnx/operators/onnx__SequenceLength.html "ONNX SequenceLength"
[159]: https://onnx.ai/onnx/operators/onnx__SequenceMap.html "ONNX SequenceMap"
[160]: https://onnx.ai/onnx/operators/onnx__Shape.html "ONNX Shape"
[161]: https://onnx.ai/onnx/operators/onnx__Shrink.html "ONNX Shrink"
[162]: https://onnx.ai/onnx/operators/onnx__Sigmoid.html "ONNX Sigmoid"
[163]: https://onnx.ai/onnx/operators/onnx__Sign.html "ONNX Sign"
[164]: https://onnx.ai/onnx/operators/onnx__Sin.html "ONNX Sin"
[165]: https://onnx.ai/onnx/operators/onnx__Sinh.html "ONNX Sinh"
[166]: https://onnx.ai/onnx/operators/onnx__Size.html "ONNX Size"
[167]: https://onnx.ai/onnx/operators/onnx__Slice.html "ONNX Slice"
[168]: https://onnx.ai/onnx/operators/onnx__Softmax.html "ONNX Softmax"
[169]: https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html "ONNX SoftmaxCrossEntropyLoss"
[170]: https://onnx.ai/onnx/operators/onnx__Softplus.html "ONNX Softplus"
[171]: https://onnx.ai/onnx/operators/onnx__Softsign.html "ONNX Softsign"
[172]: https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html "ONNX SpaceToDepth"
[173]: https://onnx.ai/onnx/operators/onnx__Split.html "ONNX Split"
[174]: https://onnx.ai/onnx/operators/onnx__SplitToSequence.html "ONNX SplitToSequence"
[175]: https://onnx.ai/onnx/operators/onnx__Sqrt.html "ONNX Sqrt"
[176]: https://onnx.ai/onnx/operators/onnx__Squeeze.html "ONNX Squeeze"
[177]: https://onnx.ai/onnx/operators/onnx__STFT.html "ONNX STFT"
[178]: https://onnx.ai/onnx/operators/onnx__StringNormalizer.html "ONNX StringNormalizer"
[179]: https://onnx.ai/onnx/operators/onnx__Sub.html "ONNX Sub"
[180]: https://onnx.ai/onnx/operators/onnx__Sum.html "ONNX Sum"
[181]: https://onnx.ai/onnx/operators/onnx__Tan.html "ONNX Tan"
[182]: https://onnx.ai/onnx/operators/onnx__Tanh.html "ONNX Tanh"
[183]: https://onnx.ai/onnx/operators/onnx__TfIdfVectorizer.html "ONNX TfIdfVectorizer"
[184]: https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html "ONNX ThresholdedRelu"
[185]: https://onnx.ai/onnx/operators/onnx__Tile.html "ONNX Tile"
[186]: https://onnx.ai/onnx/operators/onnx__TopK.html "ONNX TopK"
[187]: https://onnx.ai/onnx/operators/onnx__Transpose.html "ONNX Transpose"
[188]: https://onnx.ai/onnx/operators/onnx__Trilu.html "ONNX Trilu"
[189]: https://onnx.ai/onnx/operators/onnx__Unique.html "ONNX Unique"
[190]: https://onnx.ai/onnx/operators/onnx__Upsample.html "ONNX Upsample"
[191]: https://onnx.ai/onnx/operators/onnx__Where.html "ONNX Where"
[192]: https://onnx.ai/onnx/operators/onnx__Xor.html "ONNX Xor"
[193]: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html "ONNX Unsqueeze"
