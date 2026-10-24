//! CPU-only worker tests: the finishing layer's byte
//! parity with the Python reference postprocessor for all three question
//! types (recorded once from `models.valen.postprocess` in the pinned venv
//! and pinned here), the head readout's formula on a deterministic synthetic
//! head, and the HTTP surface over real sockets against a stub inference —
//! the same shape as the reference suite's `test_worker_server.py`
//! (FakeProcessor/FakeExecutor), with the real wire and finishing layers in
//! the loop.

use std::sync::Arc;
use std::time::Duration;

use omni_valen_native::finish::{
    answer, build_response, choice_answer, json_string, noul_answer, python_float, score_answer,
};
use omni_valen_native::protocol::{self, Kind};
use omni_valen_native::server::{Inference, Worker};
use omni_valen_native::wire::RequestError;

// ---------------------------------------------------------------------------
// Finishing parity — the bytes the reference worker sends for these logits.
// ---------------------------------------------------------------------------

/// Recorded from the pinned venv's `repr` (CPython 3.12): shortest round-trip
/// digits, the fixed/scientific switch at 1e-4 and 1e16, two-digit exponents.
#[test]
fn float_formatting_matches_cpython_repr() {
    let vectors: &[(&str, &str)] = &[
        ("0", "0.0"),
        ("-0", "-0.0"),
        ("1", "1.0"),
        ("0.5", "0.5"),
        ("0.2689414213699951", "0.2689414213699951"),
        ("0.7310585786300049", "0.7310585786300049"),
        ("0.0001", "0.0001"),
        ("0.00001", "1e-05"),
        ("0.000061", "6.1e-05"),
        ("1e15", "1000000000000000.0"),
        ("1e16", "1e+16"),
        ("1e-300", "1e-300"),
        ("5e-324", "5e-324"),
        ("2.2250738585072014e-308", "2.2250738585072014e-308"),
        ("1.7976931348623157e308", "1.7976931348623157e+308"),
        ("123.456", "123.456"),
        ("100", "100.0"),
        ("0.1", "0.1"),
        ("0.3333333333333333", "0.3333333333333333"),
        ("3", "3.0"),
        ("9.5e-7", "9.5e-07"),
        ("0.9999999999999999", "0.9999999999999999"),
        ("2", "2.0"),
        ("4", "4.0"),
    ];
    for (literal, expected) in vectors {
        let value: f64 = literal.parse().unwrap();
        assert_eq!(&python_float(value), expected, "repr of {literal}");
    }
    // The exact shortest-decimal tie bit patterns pinned by the shared
    // `tests/qwen3_5/json.rs` oracle: at a tie Rust's own `{:e}` picks the
    // upper digit (…0.3) where CPython's repr picks the even one (…0.2), so
    // the finisher delegates to the shared float_repr instead of deriving
    // from `{:e}` itself.
    for (bits, expected) in [
        (0x430c_6bf5_2634_0002u64, "1000000000000000.2"),
        (0xc30c_6bf5_2634_0002, "-1000000000000000.2"),
        (0x4315_dccb_2cbf_5325, "1538434924532937.2"),
        (0xc2ec_82d6_25e7_6314, "-250786113075992.62"),
        (0x42ed_db24_4866_3844, "262616350142914.12"),
    ] {
        assert_eq!(python_float(f64::from_bits(bits)), expected, "{bits:016x}");
    }
}

#[test]
fn json_strings_match_the_reference_escaping() {
    assert_eq!(json_string("plain"), "\"plain\"");
    assert_eq!(json_string("say \"hi\""), "\"say \\\"hi\\\"\"");
    assert_eq!(json_string("back\\slash"), "\"back\\\\slash\"");
    assert_eq!(json_string("a\nb\tc"), "\"a\\nb\\tc\"");
    assert_eq!(
        json_string("\u{1}\u{8}\u{c}\u{1f}"),
        "\"\\u0001\\b\\f\\u001f\""
    );
    // ensure_ascii=False leaves non-ASCII verbatim, including the noul wording.
    assert_eq!(
        json_string("True / 是：满足问题中的条件。"),
        "\"True / 是：满足问题中的条件。\""
    );
}

/// Recorded from the pinned venv's `models.valen.postprocess` + `json.dumps`:
/// the exact response body bytes for these keys and logits.
#[test]
fn choice_responses_are_byte_identical_to_the_reference() {
    struct Case {
        keys: &'static [&'static str],
        logits: &'static [f32],
        body: &'static str,
    }
    let cases = [
        Case {
            keys: &["ok"],
            logits: &[0.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"ok\", \"probabilities\": {\"ok\": 1.0}, \"confidence\": 1.0}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["down", "up"],
            logits: &[0.0, 1.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"up\", \"probabilities\": {\"down\": 0.2689414213699951, \"up\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["a", "b", "c"],
            logits: &[0.5, 0.2, 0.3],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"a\", \"probabilities\": {\"a\": 0.3906938314430244, \"b\": 0.2894331099035222, \"c\": 0.31987305865345345}, \"confidence\": 0.0860407471645366}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            // A tie picks the first candidate, as Python's max(range) does.
            keys: &["a", "b"],
            logits: &[1.0, 1.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"a\", \"probabilities\": {\"a\": 0.5, \"b\": 0.5}, \"confidence\": 0.0}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            // Candidate keys pass through as UTF-8, never \u-escaped.
            keys: &["白色", "黑色"],
            logits: &[0.3, 0.7],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"黑色\", \"probabilities\": {\"白色\": 0.4013123456158106, \"黑色\": 0.5986876543841895}, \"confidence\": 0.19737530876837894}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["x"],
            logits: &[2.5],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"x\", \"probabilities\": {\"x\": 1.0}, \"confidence\": 1.0}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["a", "b", "c", "d"],
            logits: &[-1.0, -0.5, -2.0, 0.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"d\", \"probabilities\": {\"a\": 0.17437148764032928, \"b\": 0.2874899806762354, \"c\": 0.06414768542935739, \"d\": 0.47399084625407806}, \"confidence\": 0.2986544616721041}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
    ];
    for case in cases {
        let answer = choice_answer(
            &case.keys.iter().map(|k| k.to_string()).collect::<Vec<_>>(),
            case.logits,
        )
        .unwrap();
        let body = build_response(&[("q".to_string(), answer)], 17, 42).unwrap();
        assert_eq!(body, case.body.as_bytes(), "keys {:?}", case.keys);
    }
    // Several questions keep the request's insertion order through the map.
    let a = choice_answer(&["p".into(), "q".into()], &[0.0, 1.0]).unwrap();
    let b = choice_answer(&["z".into()], &[0.0]).unwrap();
    let body = build_response(&[("second".into(), a), ("first".into(), b)], 5, 6).unwrap();
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("\"answers\": {\"second\":"));
    assert!(text.contains(", \"first\":"));
}

/// The reference runs on CPython 3.12, whose float `sum()` is Neumaier's
/// compensated accumulation instead of the naive left fold of earlier
/// versions, so the softmax denominator and the score sums must round exactly
/// as that accumulation does. Recorded from the pinned venv's
/// `models.valen.postprocess` + `json.dumps` for inputs where the compensated
/// and naive reductions disagree: the review's exact FP32 `[0, -1, …, -9]`
/// (reference denominator 1.5819048852379487 where the naive fold gives
/// 1.5819048852379485, so P₀ is …66, not …67), a 255-option descent at the
/// wire maximum, an f32-domain softmax divergence, and a score whose expected
/// level only matches under the compensated `sum(i·p)`.
#[test]
fn equal_logit_reductions_match_cpythons_compensated_sum() {
    assert_eq!(
        choice_answer(
            &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"].map(String::from),
            &[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
        )
        .unwrap(),
        "{\"type\": \"choice\", \"choice\": \"a\", \"probabilities\": {\"a\": 0.6321492583604866, \"b\": 0.23255471590259755, \"c\": 0.08555209892803112, \"d\": 0.031472858344688034, \"e\": 0.0115782175399118, \"f\": 0.004259388198344144, \"g\": 0.0015669413501390806, \"h\": 0.0005764455082375902, \"i\": 0.00021206245143623277, \"j\": 7.801341612780744e-05}, \"confidence\": 0.591276953733874}",
    );

    let keys: Vec<String> = (0..255).map(|i| format!("k{i}")).collect();
    let logits: Vec<f32> = (0..255).map(|i| -(i as f32)).collect();
    assert_eq!(
        choice_answer(&keys, &logits).unwrap(),
        CHOICE_255_DESCENT_ORACLE
    );

    assert_eq!(
        choice_answer(
            &["c0", "c1", "c2", "c3"].map(String::from),
            &[-5.131, 0.431, -1.612, -5.304],
        )
        .unwrap(),
        "{\"type\": \"choice\", \"choice\": \"c1\", \"probabilities\": {\"c0\": 0.0033791225629899607, \"c1\": 0.8797309440148803, \"c2\": 0.11404762592805459, \"c3\": 0.0028423074940750748}, \"confidence\": 0.8396412586865071}",
    );

    assert_eq!(
        score_answer(
            &(0..8).map(|i| i.to_string()).collect::<Vec<_>>(),
            &(0..8).map(|i| format!("D{i}")).collect::<Vec<_>>(),
            &[-0.577, -0.952, 5.479, 0.306, 2.245, -0.004, 2.599, -0.683],
        )
        .unwrap(),
        "{\"type\": \"score\", \"probabilities\": {\"0\": 0.002108792441443368, \"1\": 0.0014493504361994655, \"2\": 0.8997487831383824, \"3\": 0.005099362323165195, \"4\": 0.035449721206070595, \"5\": 0.00374011179530084, \"6\": 0.05050717877168528, \"7\": 0.0018966998877529488}, \"score\": 2.293064419327629, \"legend\": {\"0\": \"D0\", \"1\": \"D1\", \"2\": \"D2\", \"3\": \"D3\", \"4\": \"D4\", \"5\": \"D5\", \"6\": \"D6\", \"7\": \"D7\"}, \"confidence\": 0.8478008550170995}",
    );
}

/// `json.dumps(models.valen.postprocess.answer(...))` for the 255-option
/// descent `logits[i] = -i`: long enough that the compensated and naive
/// denominators diverge mid-sequence (the total is 1.5819767068693265).
const CHOICE_255_DESCENT_ORACLE: &str = r#"{"type": "choice", "choice": "k0", "probabilities": {"k0": 0.6321205588285577, "k1": 0.23254415793482963, "k2": 0.08554821486874875, "k3": 0.031471429479129766, "k4": 0.011577691889648712, "k5": 0.0042591948224191085, "k6": 0.0015668702111118421, "k7": 0.0005764193376520043, "k8": 0.0002120528238158323, "k9": 7.80098743241947e-05, "k10": 2.8698228972239195e-05, "k11": 1.0557488436917449e-05, "k12": 3.8838829463471556e-06, "k13": 1.4288006878774864e-06, "k14": 5.256263986017421e-07, "k15": 1.9336714578256666e-07, "k16": 7.113579753140744e-08, "k17": 2.616939744313904e-08, "k18": 9.627183307175361e-09, "k19": 3.54164281509871e-09, "k20": 1.3028975796473672e-09, "k21": 4.793092335042982e-10, "k22": 1.7632801296987357e-10, "k23": 6.486745087422792e-11, "k24": 2.3863401577826956e-11, "k25": 8.778854836900696e-12, "k26": 3.2295602115242415e-12, "k27": 1.188088805845063e-12, "k28": 4.3707344595632803e-13, "k29": 1.6079033504929054e-13, "k30": 5.915145860370198e-14, "k31": 2.176060553560559e-14, "k32": 8.005279403990778e-15, "k33": 2.9449777135613847e-15, "k34": 1.083396755527314e-15, "k35": 3.9855939299034196e-16, "k36": 1.466218067669163e-16, "k37": 5.393914833696036e-17, "k38": 1.984310374746451e-17, "k39": 7.299869917724197e-18, "k40": 2.6854720659566e-18, "k41": 9.879299629056328e-19, "k42": 3.6343912267024794e-19, "k43": 1.3370178134777008e-19, "k44": 4.918613660584403e-20, "k45": 1.8094568447940125e-20, "k46": 6.656619728866625e-21, "k47": 2.448833545946252e-21, "k48": 9.008755164045887e-22, "k49": 3.314135815399546e-22, "k50": 1.2192024317354473e-22, "k51": 4.485195092616999e-23, "k52": 1.650011064216837e-23, "k53": 6.0700514823078684e-24, "k54": 2.233047147193304e-24, "k55": 8.214921366189561e-25, "k56": 3.022100681461157e-25, "k57": 1.1117687098597656e-25, "k58": 4.0899685169510595e-26, "k59": 1.5046153324247487e-26, "k60": 5.535170476704004e-27, "k61": 2.036275421758535e-27, "k62": 7.491038642276729e-28, "k63": 2.755799109514443e-28, "k64": 1.0138018363889318e-28, "k65": 3.729568530293422e-29, "k66": 1.3720315867349416e-29, "k67": 5.0474221339761756e-30, "k68": 1.8568428340035245e-30, "k69": 6.830943041164138e-31, "k70": 2.5129635086574157e-31, "k71": 9.244676112491171e-32, "k72": 3.400926282074234e-32, "k73": 1.2511308601147402e-32, "k74": 4.6026532165135656e-33, "k75": 1.693221493196952e-33, "k76": 6.229013766967698e-34, "k77": 2.291526103641298e-34, "k78": 8.430053424373333e-35, "k79": 3.101243342803865e-35, "k80": 1.1408836678873418e-35, "k81": 4.197076461840207e-36, "k82": 1.5440181433355896e-36, "k83": 5.680125317288647e-37, "k84": 2.0896013275079088e-37, "k85": 7.687213686347135e-38, "k86": 2.827967875098847e-38, "k87": 1.0403512415421551e-38, "k88": 3.827238333605442e-39, "k89": 1.4079622993966921e-39, "k90": 5.1796038389251405e-40, "k91": 1.9054697657532383e-40, "k92": 7.009831525943804e-41, "k93": 2.5787729044701655e-41, "k94": 9.486775350045416e-42, "k95": 3.4899896142937215e-42, "k96": 1.2838954290005119e-42, "k97": 4.723187329432775e-43, "k98": 1.7375635152997662e-43, "k99": 6.392138950083649e-44, "k100": 2.351536504846983e-44, "k101": 8.650819352973547e-45, "k102": 3.1824585892470066e-45, "k103": 1.1707610873634455e-45, "k104": 4.306989345645345e-46, "k105": 1.5844528336073655e-46, "k106": 5.828876229899861e-47, "k107": 2.1443237301130638e-47, "k108": 7.888526155246567e-48, "k109": 2.9020265936584134e-48, "k110": 1.0675959215397215e-48, "k111": 3.9274659101294374e-49, "k112": 1.4448339642383073e-49, "k113": 5.315247113495082e-50, "k114": 1.9553701378006929e-50, "k115": 7.19340473577445e-51, "k116": 2.6463057143167112e-51, "k117": 9.735214673516262e-52, "k118": 3.581385333777188e-52, "k119": 1.3175180352095514e-52, "k120": 4.846877985261864e-53, "k121": 1.7830667646443006e-53, "k122": 6.559536049487171e-54, "k123": 2.4131184562292707e-54, "k124": 8.877366691581177e-55, "k125": 3.265800697572859e-55, "k126": 1.2014209356004099e-55, "k127": 4.419780624003502e-56, "k128": 1.625946426058777e-56, "k129": 5.981522625932068e-57, "k130": 2.2004792009822272e-57, "k131": 8.095110587667237e-58, "k132": 2.978024759212049e-58, "k133": 1.0955540842136476e-58, "k134": 4.03031824273608e-59, "k135": 1.4826712228808183e-59, "k136": 5.454442609143745e-60, "k137": 2.0065772989535048e-60, "k138": 7.381785354063174e-61, "k139": 2.7156070709002984e-61, "k140": 9.990160116840191e-62, "k141": 3.6751745209964e-62, "k142": 1.3520211489916788e-62, "k143": 4.973807847430302e-63, "k144": 1.829761651406794e-63, "k145": 6.731316937964668e-64, "k146": 2.476313113486306e-64, "k147": 9.109846843548567e-65, "k148": 3.3513253659620747e-65, "k149": 1.2328837028138075e-65, "k150": 4.535525676205221e-66, "k151": 1.6685266511811047e-66, "k152": 6.138166520161628e-67, "k153": 2.2581052692543164e-67, "k154": 8.307105045595674e-68, "k155": 3.0560131619262053e-68, "k156": 1.124244414221985e-68, "k157": 4.1358640684409936e-69, "k158": 1.5214993622591203e-69, "k159": 5.5972833513059114e-70, "k160": 2.0591254713566362e-70, "k161": 7.575099277045623e-71, "k162": 2.78672328885774e-71, "k163": 1.0251782062044293e-71, "k164": 3.771419855996271e-72, "k165": 1.3874278290467897e-72, "k166": 5.1040617441544037e-73, "k167": 1.8776793821440594e-73, "k168": 6.9075964180229565e-74, "k169": 2.541162710100142e-74, "k170": 9.34841517717348e-75, "k171": 3.43908975121721e-75, "k172": 1.2651704158162217e-75, "k173": 4.6543018555711296e-76, "k174": 1.7122219656707142e-76, "k175": 6.298912598924109e-77, "k176": 2.3172404468799586e-77, "k177": 8.524651206580624e-78, "k178": 3.136043922058342e-78, "k179": 1.153686085535921e-78, "k180": 4.2441739243422344e-79, "k181": 1.5613443315214284e-79, "k182": 5.743864801563022e-80, "k183": 2.113049773363322e-80, "k184": 7.773475697923417e-81, "k185": 2.8597018957118547e-81, "k186": 1.0520255353113912e-81, "k187": 3.8701856602844207e-82, "k188": 1.4237617379351625e-82, "k189": 5.23772672512869e-83, "k190": 1.9268519806490713e-83, "k191": 7.0884922986126715e-84, "k192": 2.6077105855617024e-84, "k193": 9.593231129532935e-85, "k194": 3.529152506961061e-85, "k195": 1.2983026520696298e-85, "k196": 4.7761885411477694e-86, "k197": 1.7570615714468876e-86, "k198": 6.463868290076973e-87, "k199": 2.3779242543593234e-87, "k200": 8.747894458417265e-88, "k201": 3.218170524789301e-88, "k202": 1.1838987742538953e-88, "k203": 4.3553201947607856e-89, "k204": 1.602232759371295e-89, "k205": 5.894284921440899e-90, "k206": 2.168386243004937e-90, "k207": 7.977047193204996e-91, "k208": 2.934591663634476e-91, "k209": 1.0795759412842245e-91, "k210": 3.971537939817743e-92, "k211": 1.4610471578913326e-92, "k212": 5.3748921197018756e-93, "k213": 1.977312309352715e-93, "k214": 7.274125473860908e-94, "k215": 2.6760012143349037e-94, "k216": 9.844458313036257e-95, "k217": 3.621573822835337e-95, "k218": 1.332302554105788e-95, "k219": 4.901267190757227e-96, "k220": 1.8030754351676933e-96, "k221": 6.633143834794462e-97, "k222": 2.4401972471539846e-97, "k223": 8.976983996310998e-98, "k224": 3.302447855967871e-98, "k225": 1.2149026717512882e-98, "k226": 4.469377159615561e-99, "k227": 1.6441919718637807e-99, "k228": 6.048644237878195e-100, "k229": 2.225171862075495e-100, "k230": 8.185949811307508e-101, "k231": 3.01144264204128e-101, "k232": 1.1078478362739977e-101, "k233": 4.075544429114698e-102, "k234": 1.4993090070521002e-102, "k235": 5.515649596576367e-103, "k236": 2.0290940912860052e-103, "k237": 7.464620003865712e-104, "k238": 2.7460802355792877e-104, "k239": 1.010226462476851e-104, "k240": 3.71641546472587e-105, "k241": 1.3671928443242592e-105, "k242": 5.029621395436032e-106, "k243": 1.8502943082569372e-106, "k244": 6.806852361242626e-107, "k245": 2.50410104279045e-107, "k246": 9.212072922585766e-108, "k247": 3.3889322387914268e-108, "k248": 1.246718498174475e-108, "k249": 4.586421044065257e-109, "k250": 1.6872500106676698e-109, "k251": 6.207045910409325e-110, "k252": 2.283444580846869e-110, "k253": 8.400323163479045e-111, "k254": 3.0903061910401938e-111}, "confidence": 0.6306722145719772}"#;

/// Recorded from the pinned venv's `models.valen.postprocess._noul_answer` +
/// `json.dumps`: the exact answer bytes for `true`/`false` keys and these
/// logits. P(true) plus the full pair, choice confidence on the two
/// candidates, field order type/noul/probabilities/confidence.
#[test]
fn noul_responses_are_byte_identical_to_the_reference() {
    struct Case {
        logits: &'static [f32],
        body: &'static str,
    }
    let cases = [
        // A tie: P(true) 0.5, confidence 0.
        Case {
            logits: &[0.0, 0.0],
            body: "{\"type\": \"noul\", \"noul\": 0.5, \"probabilities\": {\"true\": 0.5, \"false\": 0.5}, \"confidence\": 0.0}",
        },
        Case {
            logits: &[0.0, 1.0],
            body: "{\"type\": \"noul\", \"noul\": 0.2689414213699951, \"probabilities\": {\"true\": 0.2689414213699951, \"false\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}",
        },
        Case {
            logits: &[2.5, -3.0],
            body: "{\"type\": \"noul\", \"noul\": 0.995929862284104, \"probabilities\": {\"true\": 0.995929862284104, \"false\": 0.004070137715896127}, \"confidence\": 0.9918597245682079}",
        },
        // Negative logits pick the same softmax shape as [0,1].
        Case {
            logits: &[-1.0, -2.0],
            body: "{\"type\": \"noul\", \"noul\": 0.7310585786300049, \"probabilities\": {\"true\": 0.7310585786300049, \"false\": 0.2689414213699951}, \"confidence\": 0.4621171572600098}",
        },
    ];
    let keys: Vec<String> = ["true", "false"].iter().map(|k| k.to_string()).collect();
    for case in cases {
        let produced = noul_answer(&keys, case.logits).unwrap();
        assert_eq!(produced, case.body, "logits {:?}", case.logits);
    }
    // A logit-count mismatch is the worker's inference failure (500), as the
    // reference's ValueError on `answer` is.
    assert!(noul_answer(&keys, &[0.0]).is_err());
}

/// Recorded from the pinned venv's `models.valen.postprocess._score_answer` +
/// `json.dumps`: expected level `sum(i*p)`, the static legend keyed "0"…, and
/// the concentration confidence `max(0, 1 - distance/uniform_mad)`, field
/// order type/probabilities/score/legend/confidence.
#[test]
fn score_responses_are_byte_identical_to_the_reference() {
    struct Case {
        levels: &'static [&'static str],
        logits: &'static [f32],
        body: &'static str,
    }
    const THREE: &[&str] = &["No error", "A minor error", "A major error"];
    let cases = [
        // Uniform: score 1.0, no concentration.
        Case {
            levels: THREE,
            logits: &[0.0, 0.0, 0.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.3333333333333333, \"1\": 0.3333333333333333, \"2\": 0.3333333333333333}, \"score\": 1.0, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.0}",
        },
        // Fully peaked on level 1.
        Case {
            levels: THREE,
            logits: &[-100.0, 3.0, -100.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 1.8521167695179754e-45, \"1\": 1.0, \"2\": 1.8521167695179754e-45}, \"score\": 1.0, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 1.0}",
        },
        // Tilted toward p = [0.7, 0.2, 0.1]: mode 0, distance 0.4, MAD 2/3 →
        // 0.4-class. Logits are the f32 head outputs of ln(7)/ln(2)/0 (the
        // finisher's input contract), so the expected bytes were recorded
        // from the f32-rounded values.
        Case {
            levels: THREE,
            logits: &[1.9459101, std::f32::consts::LN_2, 0.0], // recorded values; the middle one is exactly ln 2 in f32
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.6999999886271213, \"1\": 0.20000000770889612, \"2\": 0.10000000366398262}, \"score\": 0.40000001503686133, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.39999997744470794}",
        },
        // Mass split across the endpoints clamps confidence to 0.
        Case {
            levels: THREE,
            logits: &[0.4054651, -50.0, 0.0], // f32-shortest of the recorded values,
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.5999999971505067, \"1\": 7.714999446815271e-23, \"2\": 0.4000000028494934}, \"score\": 0.8000000056989868, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.0}",
        },
        // Two levels are the noul pair in score clothing: expected level is
        // P(1), confidence the choice confidence.
        Case {
            levels: &["lo", "hi"],
            logits: &[0.0, 1.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.2689414213699951, \"1\": 0.7310585786300049}, \"score\": 0.7310585786300049, \"legend\": {\"0\": \"lo\", \"1\": \"hi\"}, \"confidence\": 0.4621171572600098}",
        },
        // Ten levels pin the legend's verbatim UTF-8 (never \u-escaped) and a
        // spread distribution that clamps confidence to 0.
        Case {
            levels: &[
                "等级 0：描述“0”",
                "等级 1：描述“1”",
                "等级 2：描述“2”",
                "等级 3：描述“3”",
                "等级 4：描述“4”",
                "等级 5：描述“5”",
                "等级 6：描述“6”",
                "等级 7：描述“7”",
                "等级 8：描述“8”",
                "等级 9：描述“9”",
            ],
            logits: &[0.5, 0.2, 0.3, -1.0, -2.0, 0.0, 1.0, 0.1, -0.5, 0.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.1478252054193804, \"1\": 0.10951160597706486, \"2\": 0.12102904319969529, \"3\": 0.03298426175920089, \"4\": 0.012134231783427398, \"5\": 0.08966051936517236, \"6\": 0.24372256052054833, \"7\": 0.0990901986496028, \"8\": 0.054381853960735325, \"9\": 0.08966051936517236}, \"score\": 4.345328261256573, \"legend\": {\"0\": \"等级 0：描述“0”\", \"1\": \"等级 1：描述“1”\", \"2\": \"等级 2：描述“2”\", \"3\": \"等级 3：描述“3”\", \"4\": \"等级 4：描述“4”\", \"5\": \"等级 5：描述“5”\", \"6\": \"等级 6：描述“6”\", \"7\": \"等级 7：描述“7”\", \"8\": \"等级 8：描述“8”\", \"9\": \"等级 9：描述“9”\"}, \"confidence\": 0.0}",
        },
        // A three-way tie: mode is level 0 by candidate order, uniform score 1.
        Case {
            levels: &["a", "b", "c"],
            logits: &[1.0, 1.0, 1.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.3333333333333333, \"1\": 0.3333333333333333, \"2\": 0.3333333333333333}, \"score\": 1.0, \"legend\": {\"0\": \"a\", \"1\": \"b\", \"2\": \"c\"}, \"confidence\": 0.0}",
        },
    ];
    for case in cases {
        let keys: Vec<String> = (0..case.levels.len()).map(|i| i.to_string()).collect();
        let levels: Vec<String> = case.levels.iter().map(|l| l.to_string()).collect();
        let produced = score_answer(&keys, &levels, case.logits).unwrap();
        assert_eq!(produced, case.body, "levels {:?}", case.levels);
    }
    let keys: Vec<String> = (0..3).map(|i| i.to_string()).collect();
    let levels: Vec<String> = THREE.iter().map(|l| l.to_string()).collect();
    assert!(score_answer(&keys, &levels, &[0.0, 1.0]).is_err());
}

/// The finisher's per-kind dispatch, `postprocess.answer`'s shape: one mixed
/// body finishes choice + noul + score in request order with each type's
/// exact bytes. Expected body recorded from the pinned venv (stub logits:
/// candidate k scores k).
#[test]
fn mixed_type_bodies_finish_every_question_in_order() {
    let body = vec![
        (
            "move".to_string(),
            answer(
                Kind::Choice,
                &["down".to_string(), "wait".to_string()],
                &["Close".to_string(), "Wait".to_string()],
                &[0.0, 1.0],
            )
            .unwrap(),
        ),
        (
            "dup".to_string(),
            answer(
                Kind::Noul,
                &["true".to_string(), "false".to_string()],
                &[
                    "True / 是：满足问题中的条件。".to_string(),
                    "False / 否：不满足问题中的条件。".to_string(),
                ],
                &[0.0, 1.0],
            )
            .unwrap(),
        ),
        (
            "sev".to_string(),
            answer(
                Kind::Score,
                &["0".to_string(), "1".to_string(), "2".to_string()],
                &[
                    "No error".to_string(),
                    "A minor error".to_string(),
                    "A major error".to_string(),
                ],
                &[0.0, 1.0, 2.0],
            )
            .unwrap(),
        ),
    ];
    let bytes = build_response(&body, 17, 42).unwrap();
    assert_eq!(
        String::from_utf8(bytes).unwrap(),
        "{\"model\": \"valen-preview-0923\", \"answers\": {\"move\": {\"type\": \"choice\", \"choice\": \"wait\", \"probabilities\": {\"down\": 0.2689414213699951, \"wait\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"dup\": {\"type\": \"noul\", \"noul\": 0.2689414213699951, \"probabilities\": {\"true\": 0.2689414213699951, \"false\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"sev\": {\"type\": \"score\", \"probabilities\": {\"0\": 0.09003057317038046, \"1\": 0.24472847105479764, \"2\": 0.6652409557748218}, \"score\": 1.5752103826044412, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.3628155739066622}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
    );
}

// ---------------------------------------------------------------------------
// Head readout on the deterministic synthetic head (tests/valen/data).
// ---------------------------------------------------------------------------

fn lcg(i: usize, j: usize) -> f32 {
    ((((i.wrapping_mul(73_856_093)) ^ j.wrapping_mul(19_349_663)) & 0xFFFF_FFFF) % 1_000_003) as f32
        / 1_000_003.0
        - 0.5
}

/// The fixture's logits were recorded from numpy over the same deterministic
/// weights; the head's sequential FP32 reduction may differ from BLAS only at
/// the FP32 rounding scale.
#[test]
fn head_readout_matches_the_recorded_numpy_logits() {
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../../tests/valen/data/head-readout.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(fixture["format"], "valen-head-readout/1");
    let (projection, hidden) = (256usize, 2048usize);
    let candidates = fixture["candidates"].as_u64().unwrap() as usize;
    let candidate: Vec<f32> = (0..projection)
        .flat_map(|i| (0..hidden).map(move |j| lcg(i, j)))
        .collect();
    let decision: Vec<f32> = (0..projection)
        .flat_map(|i| (0..hidden).map(move |j| lcg(i + 7919, j + 104_729)))
        .collect();
    let rows: Vec<Vec<f32>> = (0..candidates)
        .map(|k| (0..hidden).map(|j| lcg(3 + k, 31 + j)).collect())
        .collect();
    let hidden_decision: Vec<f32> = (0..hidden).map(|j| lcg(5, 7 + j)).collect();
    let head = omni_valen_native::head::Head::from_parts(candidate, decision, projection, hidden);
    let logits = head.logits(&rows, &hidden_decision).unwrap();
    let recorded_f32: Vec<f64> = fixture["logits_f32"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let recorded_f64: Vec<f64> = fixture["logits_f64"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(logits.len(), recorded_f32.len());
    for (index, value) in logits.iter().enumerate() {
        let delta = (*value as f64 - recorded_f32[index]).abs();
        assert!(
            delta <= 1e-3,
            "logit {index}: native {value} vs numpy f32 {} (delta {delta})",
            recorded_f32[index]
        );
        assert!(
            (*value as f64 - recorded_f64[index]).abs() <= 1e-3,
            "logit {index} vs the float64 oracle"
        );
    }
}

// -----------------------------------------------------------------------
// HTTP contract over real sockets, stub inference (no tokenizer, no CUDA).
// -----------------------------------------------------------------------

mod http {
    use super::*;

    /// The reference suite's FakeProcessor/FakeExecutor shape: the real wire
    /// layer and the real finishing over fixed logits.
    struct Stub {
        delay: Duration,
        panic: bool,
    }

    impl Inference for Stub {
        fn decide(&self, raw: &[u8]) -> Result<Vec<u8>, RequestError> {
            if self.panic {
                panic!("executor boom");
            }
            if self.delay > Duration::ZERO {
                std::thread::sleep(self.delay);
            }
            let body = omni_valen_native::wire::parse_body(raw)?;
            let request = protocol::parse_request(&body, &protocol::default_special_tokens())?;
            let mut answers = Vec::new();
            for question in &request.questions {
                let logits: Vec<f32> = question
                    .keys
                    .iter()
                    .enumerate()
                    .map(|(i, _)| i as f32)
                    .collect();
                let descriptions: Vec<String> = question
                    .descriptions
                    .iter()
                    .map(|d| d.clone().unwrap_or_default())
                    .collect();
                let produced = answer(question.kind, &question.keys, &descriptions, &logits)
                    .map_err(|_| omni_valen_native::wire::error(500, "inference failed"))?;
                answers.push((question.name.clone(), produced));
            }
            // Token accounting values are the stub's; the compiled accounting
            // is pinned by the compiler tests and the GPU gate.
            build_response(&answers, 17, 17)
                .map_err(|_| omni_valen_native::wire::error(500, "inference failed"))
        }
    }

    fn body() -> Vec<u8> {
        br#"{"model": "valen-preview-0923", "state": "The card was charged twice.", "questions": {"move": {"type": "choice", "instructions": "Close it?", "criteria": {"down": "Close", "wait": "Wait"}}}}"#.to_vec()
    }

    /// A choice + noul + score body (the all-types round trip): the same
    /// stub-logits convention, so the response is the recorded reference body.
    fn mixed_body() -> Vec<u8> {
        br#"{"model": "valen-preview-0923", "state": "The card was charged twice.", "questions": {"move": {"type": "choice", "instructions": "Close it?", "criteria": {"down": "Close", "wait": "Wait"}}, "dup": {"type": "noul", "instructions": "Was it duplicated?"}, "sev": {"type": "score", "instructions": "How severe?", "criteria": ["No error", "A minor error", "A major error"]}}}"#.to_vec()
    }

    async fn start(worker: Worker) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(
                listener,
                omni_valen_native::server::router(Arc::new(worker)),
            )
            .await
            .unwrap();
        });
        url
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn health_and_unknown_routes_match_the_reference_bodies() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let health = client().get(format!("{url}/health")).send().await.unwrap();
        assert_eq!(health.status(), 200);
        assert_eq!(health.headers()["content-type"], "application/json");
        assert_eq!(
            health.bytes().await.unwrap(),
            b"{\"status\": \"ready\", \"model\": \"valen-preview-0923\", \"modality\": \"multimodal\"}"[..]
        );
        let get = client().get(format!("{url}/nope")).send().await.unwrap();
        assert_eq!(get.status(), 404);
        assert_eq!(
            get.bytes().await.unwrap(),
            b"{\"detail\": \"unknown route\"}"[..]
        );
        let post = client()
            .post(format!("{url}/other"))
            .header("content-type", "application/json")
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(post.status(), 404);
        assert_eq!(
            post.bytes().await.unwrap(),
            b"{\"detail\": \"unknown route\"}"[..]
        );
    }

    #[tokio::test]
    async fn a_valid_choice_request_answers_200() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let response = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["content-type"], "application/json");
        let text = String::from_utf8(response.bytes().await.unwrap().to_vec()).unwrap();
        assert_eq!(
            text,
            "{\"model\": \"valen-preview-0923\", \"answers\": {\"move\": {\"type\": \"choice\", \"choice\": \"wait\", \"probabilities\": {\"down\": 0.2689414213699951, \"wait\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 17}}",
        );
    }

    #[tokio::test]
    async fn client_errors_keep_the_reference_status_and_detail() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let post = |payload: Vec<u8>| {
            let client = client();
            let url = url.clone();
            async move {
                client
                    .post(format!("{url}/v1/systemone"))
                    .header("content-type", "application/json")
                    .body(payload)
                    .send()
                    .await
                    .unwrap()
            }
        };
        let response = post(b"{not json".to_vec()).await;
        assert_eq!(response.status(), 400);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"request body must contain valid JSON and UTF-8 text\"}"[..]
        );

        let wrong_model = br#"{"model": "Valen", "state": "x", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A"}}}}"#.to_vec();
        let response = post(wrong_model).await;
        assert_eq!(response.status(), 422);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"model must be valen-preview-0923\"}"[..]
        );

        // The interim milestone guard is removed: all three question types serve.
        // A malformed noul still gets the wire layer's 422, byte-identical to
        // the reference's validation order.
        let bad_noul = br#"{"model": "valen-preview-0923", "state": "x", "questions": {"n": {"type": "noul", "instructions": "i", "criteria": {"maybe": "Neither"}}}}"#.to_vec();
        let response = post(bad_noul).await;
        assert_eq!(response.status(), 422);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"question 'n': noul criteria must describe true and/or false\"}"[..]
        );

        let bad_score = br#"{"model": "valen-preview-0923", "state": "x", "questions": {"s": {"type": "score", "instructions": "i", "criteria": ["only one"]}}}"#.to_vec();
        let response = post(bad_score).await;
        assert_eq!(response.status(), 422);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"question 's': score requires an ordered list of 2 to 10 level descriptions\"}"[..]
        );
    }

    /// All three question types in one body answer 200 through the real wire
    /// and finishing layers, each type's bytes the reference's own.
    #[tokio::test]
    async fn a_mixed_type_request_answers_200_for_every_kind() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let response = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(mixed_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let text = String::from_utf8(response.bytes().await.unwrap().to_vec()).unwrap();
        assert_eq!(
            text,
            "{\"model\": \"valen-preview-0923\", \"answers\": {\"move\": {\"type\": \"choice\", \"choice\": \"wait\", \"probabilities\": {\"down\": 0.2689414213699951, \"wait\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"dup\": {\"type\": \"noul\", \"noul\": 0.2689414213699951, \"probabilities\": {\"true\": 0.2689414213699951, \"false\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"sev\": {\"type\": \"score\", \"probabilities\": {\"0\": 0.09003057317038046, \"1\": 0.24472847105479764, \"2\": 0.6652409557748218}, \"score\": 1.5752103826044412, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.3628155739066622}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 17}}",
        );
    }

    #[tokio::test]
    async fn header_gates_match_the_reference_order_and_bodies() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let client = client();
        // 415 before the body is read, whatever the body says.
        let response = client
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "text/plain")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 415);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"Content-Type must be application/json\"}"[..]
        );
        // Parameters do not change the media type.
        let response = client
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json; charset=utf-8")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    /// Raw-socket cases axum's own client cannot produce: chunked upload,
    /// duplicate Content-Length, oversized declared length, truncated body
    /// and the slow-body timeout.
    #[tokio::test]
    async fn raw_sockets_reproduce_the_reference_transport_rejections() {
        let url = start(
            Worker::new(Arc::new(Stub {
                delay: Duration::ZERO,
                panic: false,
            }))
            .with_body_timeout(Duration::from_millis(300)),
        )
        .await;
        let address = Arc::new(url.trim_start_matches("http://").to_owned());
        let raw = |request: String| {
            let address = address.clone();
            async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut stream = tokio::net::TcpStream::connect(&*address).await.unwrap();
                stream.write_all(request.as_bytes()).await.unwrap();
                let mut buffer = Vec::new();
                let _ = stream.read_to_end(&mut buffer).await;
                buffer
            }
        };
        let payload = body();

        // Chunked: the reference rejects Transfer-Encoding with 411.
        let response = raw(format!(
            "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n{:x}\r\n",
            payload.len()
        ))
        .await;
        let head = String::from_utf8_lossy(&response).to_string();
        assert!(
            head.starts_with("HTTP/1.1 411")
                && head.contains("Content-Length is required; chunked requests are unsupported"),
            "chunked: {head}"
        );

        // Duplicate Content-Length is unreachable through hyper: it normalizes
        // equal duplicates (and rejects conflicting ones) before the handler
        // runs, where the reference answers 411 "one Content-Length is
        // required" — a recorded transport deviation, see native.md.

        // Declared length beyond the limit: 413 before reading the body.
        let response = raw(
            "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 9000000\r\n\r\n"
                .to_owned(),
        )
        .await;
        assert!(
            response.starts_with(b"HTTP/1.1 413"),
            "{}",
            String::from_utf8_lossy(&response)
        );
        assert!(response.ends_with(b"{\"detail\": \"request exceeds body limit\"}"));

        // Truncated body with the write side closed: 400 incomplete body
        // (a stall without closing is the 408 case below, as in Python).
        {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut stream = tokio::net::TcpStream::connect(&*address).await.unwrap();
            stream
                .write_all(
                    format!(
                        "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{{",
                        payload.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            stream.shutdown().await.unwrap();
            let mut buffer = Vec::new();
            let _ = stream.read_to_end(&mut buffer).await;
            let head = String::from_utf8_lossy(&buffer).to_string();
            assert!(
                head.starts_with("HTTP/1.1 400") && head.contains("incomplete body"),
                "truncated body: {head}"
            );
        }

        // A body that stalls past the timeout: 408.
        use tokio::io::AsyncWriteExt;
        let mut stream = tokio::net::TcpStream::connect(&*address).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    payload.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.flush().await.unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
        let mut buffer = Vec::new();
        use tokio::io::AsyncReadExt;
        let _ = stream.read_to_end(&mut buffer).await;
        let head = String::from_utf8_lossy(&buffer).to_string();
        assert!(
            head.starts_with("HTTP/1.1 408") && head.contains("request body timed out"),
            "slow body: {head}"
        );
    }

    #[tokio::test]
    async fn busy_worker_returns_503_and_keeps_serving() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::from_millis(600),
            panic: false,
        })))
        .await;
        let first = {
            let client = client();
            let url = url.clone();
            tokio::spawn(async move {
                client
                    .post(format!("{url}/v1/systemone"))
                    .header("content-type", "application/json")
                    .body(body())
                    .send()
                    .await
                    .unwrap()
                    .status()
            })
        };
        tokio::time::sleep(Duration::from_millis(150)).await;
        let busy = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(busy.status(), 503);
        assert_eq!(
            busy.bytes().await.unwrap(),
            b"{\"detail\": \"worker busy\"}"[..]
        );
        assert_eq!(first.await.unwrap(), 200);
        // Admission is released after completion.
        let after = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(after.status(), 200);
    }

    #[tokio::test]
    async fn a_panicking_inference_maps_to_500_inference_failed() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: true,
        })))
        .await;
        let response = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 500);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"inference failed\"}"[..]
        );
    }

    /// A panic inside the *guarded* decision poisons the mutex. The reference
    /// holds its lock in a Python `with` block, which simply releases on the
    /// way out — the failing request answers 500 and the next one proceeds on
    /// a fresh lock. The native worker must recover the poison the same way
    /// instead of turning every later request into a permanent 500.
    struct Flaky {
        calls: usize,
    }

    impl omni_valen_native::server::Decide for Flaky {
        fn decide(
            &mut self,
            _raw: &[u8],
        ) -> Result<omni_valen_native::engine::Outcome, RequestError> {
            self.calls += 1;
            if self.calls == 1 {
                panic!("engine boom");
            }
            Ok(omni_valen_native::engine::Outcome {
                answers: Vec::new(),
                logical_tokens: 0,
                compute_tokens: 0,
            })
        }
    }

    #[tokio::test]
    async fn a_poisoned_engine_mutex_serves_the_next_request() {
        let url = start(Worker::new(Arc::new(std::sync::Mutex::new(Flaky {
            calls: 0,
        }))))
        .await;
        let first = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(first.status(), 500);
        assert_eq!(
            first.bytes().await.unwrap(),
            b"{\"detail\": \"inference failed\"}"[..]
        );
        let second = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(second.status(), 200);
        let text = String::from_utf8(second.bytes().await.unwrap().to_vec()).unwrap();
        assert!(text.starts_with("{\"model\": \"valen-preview-0923\", \"answers\": {}"));
    }
}
