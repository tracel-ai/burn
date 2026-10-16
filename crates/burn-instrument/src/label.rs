/// A type's own name: the last path segment, generics dropped.
///
/// What a component carries when nobody has named the role it serves. Spelled as the type is in
/// code, so an unnamed row stands out in a table of roles.
pub fn type_label<T: ?Sized>() -> &'static str {
    let name = core::any::type_name::<T>();
    let before_generics = name.split('<').next().unwrap_or(name);
    before_generics
        .rsplit("::")
        .next()
        .unwrap_or(before_generics)
}

/// A type's name as a role: [`type_label`] split into words at its capitals and lower-cased, so
/// `ShortConv` is `short conv`.
///
/// What a component is labelled when it names no label itself, which is the normal case: a
/// hand-written label equal to the type name is a second copy of it that can drift. Interned, since
/// a component is entered once per layer per pass and the answer never changes.
///
/// Without `std` there is nothing to intern into, and this is [`type_label`]; no collector can be
/// installed there to read it.
pub fn role_label<T: ?Sized>() -> &'static str {
    #[cfg(feature = "std")]
    return interned(type_label::<T>());
    #[cfg(not(feature = "std"))]
    return type_label::<T>();
}

#[cfg(feature = "std")]
fn interned(name: &'static str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static ROLES: OnceLock<Mutex<HashMap<&'static str, &'static str>>> = OnceLock::new();

    let mut roles = ROLES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Leaked once per component type, and the component types are a fixed set.
    roles
        .entry(name)
        .or_insert_with(|| Box::leak(words(name).into_boxed_str()))
}

/// `name` as lower-case words.
///
/// A capital opens a word unless it sits inside a run of them, so `RmsNorm` is `rms norm` and
/// `KVCache` is `kv cache`.
#[cfg(feature = "std")]
fn words(name: &str) -> String {
    let letters: Vec<char> = name.chars().collect();
    let mut words = String::with_capacity(name.len() + 4);
    for (at, letter) in letters.iter().enumerate() {
        let opens_word = at > 0
            && letter.is_uppercase()
            && (letters[at - 1].is_lowercase()
                || letters.get(at + 1).is_some_and(char::is_ascii_lowercase));
        if opens_word {
            words.push(' ');
        }
        words.extend(letter.to_lowercase());
    }
    words
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn a_type_label_is_the_name_without_its_path_or_generics() {
        struct Attention;
        struct Linear<T>(core::marker::PhantomData<T>);

        assert_eq!(type_label::<Attention>(), "Attention");
        assert_eq!(type_label::<Linear<u32>>(), "Linear");
    }

    #[test]
    fn a_role_label_is_the_type_name_as_words() {
        struct Attention;
        struct ShortConv;
        struct RmsNorm;
        struct GatedDeltaNet;

        assert_eq!(role_label::<Attention>(), "attention");
        assert_eq!(role_label::<ShortConv>(), "short conv");
        assert_eq!(role_label::<RmsNorm>(), "rms norm");
        assert_eq!(role_label::<GatedDeltaNet>(), "gated delta net");
    }

    #[test]
    fn a_run_of_capitals_is_one_word() {
        assert_eq!(words("KVCache"), "kv cache");
        assert_eq!(words("Rope"), "rope");
    }
}
