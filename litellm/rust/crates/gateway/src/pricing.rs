//! What the gateway prices a Messages call with: the bundled catalog, read the way the Python
//! host's cost map reads it, and each deployment's pricing identity, as Python's
//! `_pricing_identity` sends it.

use std::collections::HashSet;
use std::sync::Arc;

use litellm_cost::receipt::Receipt;
use litellm_inference_messages::{PricedCall, Pricing, PricingIdentity, price_call};
use litellm_model_catalog::{Catalog, CostLookup, KnownNames, PricingContext};
use litellm_router_core::Deployment;
use serde_json::Value;

/// The model names Python's `get_model_info` renames before it looks one up: the keys of
/// `litellm.azure_llms` and `litellm.azure_embedding_models`.
const RENAMED_MODELS: [&str; 6] = [
    "gpt-35-turbo",
    "gpt-35-turbo-instruct",
    "azure/gpt-41",
    "azure/gpt-41-mini",
    "azure/gpt-41-nano",
    "ada",
];

/// Python's `_CUSTOM_PRICING_KEYS`, the fields of `CustomPricingLiteLLMParams`, with
/// `off_peak_pricing`, which `_MODEL_INFO_CUSTOM_PRICING_KEYS` adds for `model_info`.
const CUSTOM_PRICING_KEYS: [&str; 164] = [
    "annotation_cost_per_page",
    "annotation_cost_per_page_batches",
    "cache_creation_input_audio_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_100k_tokens",
    "cache_creation_input_token_cost_above_100k_tokens_batches",
    "cache_creation_input_token_cost_above_12hr",
    "cache_creation_input_token_cost_above_12hr_above_272k_tokens",
    "cache_creation_input_token_cost_above_12hr_above_272k_tokens_batches",
    "cache_creation_input_token_cost_above_12hr_above_272k_tokens_flex",
    "cache_creation_input_token_cost_above_12hr_above_272k_tokens_priority",
    "cache_creation_input_token_cost_above_12hr_above_272k_tokens_ultrafast",
    "cache_creation_input_token_cost_above_12hr_batches",
    "cache_creation_input_token_cost_above_12hr_flex",
    "cache_creation_input_token_cost_above_12hr_priority",
    "cache_creation_input_token_cost_above_12hr_ultrafast",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_1hr_above_100k_tokens",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_creation_input_token_cost_above_200k_tokens_batches",
    "cache_creation_input_token_cost_above_272k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens_batches",
    "cache_creation_input_token_cost_above_272k_tokens_flex",
    "cache_creation_input_token_cost_above_272k_tokens_priority",
    "cache_creation_input_token_cost_above_272k_tokens_ultrafast",
    "cache_creation_input_token_cost_batches",
    "cache_creation_input_token_cost_flex",
    "cache_creation_input_token_cost_priority",
    "cache_creation_input_token_cost_ultrafast",
    "cache_read_input_audio_token_cost",
    "cache_read_input_audio_token_cost_batches",
    "cache_read_input_audio_token_cost_flex",
    "cache_read_input_audio_token_cost_priority",
    "cache_read_input_image_token_cost",
    "cache_read_input_image_token_cost_batches",
    "cache_read_input_image_token_cost_priority",
    "cache_read_input_token_cost",
    "cache_read_input_token_cost_above_100k_tokens",
    "cache_read_input_token_cost_above_100k_tokens_batches",
    "cache_read_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens_batches",
    "cache_read_input_token_cost_above_200k_tokens_priority",
    "cache_read_input_token_cost_above_272k_tokens",
    "cache_read_input_token_cost_above_272k_tokens_batches",
    "cache_read_input_token_cost_above_272k_tokens_flex",
    "cache_read_input_token_cost_above_272k_tokens_priority",
    "cache_read_input_token_cost_above_272k_tokens_ultrafast",
    "cache_read_input_token_cost_above_512k_tokens",
    "cache_read_input_token_cost_balanced",
    "cache_read_input_token_cost_batches",
    "cache_read_input_token_cost_flex",
    "cache_read_input_token_cost_priority",
    "cache_read_input_token_cost_ultrafast",
    "citation_cost_per_token",
    "cost_per_second",
    "explicit_cache_read_input_token_cost",
    "fedramp_uplift_multiplier",
    "google_maps_grounding_cost_per_query",
    "input_cost_per_audio_per_second",
    "input_cost_per_audio_per_second_above_128k_tokens",
    "input_cost_per_audio_token",
    "input_cost_per_audio_token_above_200k_tokens",
    "input_cost_per_audio_token_above_200k_tokens_batches",
    "input_cost_per_audio_token_above_200k_tokens_flex",
    "input_cost_per_audio_token_above_200k_tokens_priority",
    "input_cost_per_audio_token_batches",
    "input_cost_per_audio_token_flex",
    "input_cost_per_audio_token_priority",
    "input_cost_per_character",
    "input_cost_per_character_above_128k_tokens",
    "input_cost_per_image",
    "input_cost_per_image_above_128k_tokens",
    "input_cost_per_image_token",
    "input_cost_per_image_token_batches",
    "input_cost_per_image_token_priority",
    "input_cost_per_pixel",
    "input_cost_per_query",
    "input_cost_per_second",
    "input_cost_per_text_message",
    "input_cost_per_token",
    "input_cost_per_token_above_100k_tokens",
    "input_cost_per_token_above_100k_tokens_batches",
    "input_cost_per_token_above_128k_tokens",
    "input_cost_per_token_above_200k_tokens",
    "input_cost_per_token_above_200k_tokens_batches",
    "input_cost_per_token_above_200k_tokens_flex",
    "input_cost_per_token_above_200k_tokens_priority",
    "input_cost_per_token_above_272k_tokens",
    "input_cost_per_token_above_272k_tokens_batches",
    "input_cost_per_token_above_272k_tokens_flex",
    "input_cost_per_token_above_272k_tokens_priority",
    "input_cost_per_token_above_272k_tokens_ultrafast",
    "input_cost_per_token_above_512k_tokens",
    "input_cost_per_token_balanced",
    "input_cost_per_token_batches",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "input_cost_per_token_ultrafast",
    "input_cost_per_video_per_second",
    "input_cost_per_video_per_second_above_128k_tokens",
    "input_cost_per_video_per_second_above_15s_interval",
    "input_cost_per_video_per_second_above_8s_interval",
    "input_cost_per_video_token",
    "input_cost_per_video_token_batches",
    "ocr_cost_per_credit",
    "ocr_cost_per_page",
    "ocr_cost_per_page_batches",
    "off_peak_pricing",
    "output_cost_per_audio_per_second",
    "output_cost_per_audio_token",
    "output_cost_per_billable_unit",
    "output_cost_per_character",
    "output_cost_per_character_above_128k_tokens",
    "output_cost_per_image",
    "output_cost_per_image_1024",
    "output_cost_per_image_1536",
    "output_cost_per_image_1K",
    "output_cost_per_image_2K",
    "output_cost_per_image_512",
    "output_cost_per_image_token",
    "output_cost_per_image_token_batches",
    "output_cost_per_image_token_priority",
    "output_cost_per_pixel",
    "output_cost_per_reasoning_token",
    "output_cost_per_reasoning_token_flex",
    "output_cost_per_reasoning_token_priority",
    "output_cost_per_second",
    "output_cost_per_second_1080p",
    "output_cost_per_second_2k",
    "output_cost_per_second_480p",
    "output_cost_per_second_4k",
    "output_cost_per_second_720p",
    "output_cost_per_second_768p",
    "output_cost_per_token",
    "output_cost_per_token_above_100k_tokens",
    "output_cost_per_token_above_100k_tokens_batches",
    "output_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens_batches",
    "output_cost_per_token_above_200k_tokens_flex",
    "output_cost_per_token_above_200k_tokens_priority",
    "output_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens_batches",
    "output_cost_per_token_above_272k_tokens_flex",
    "output_cost_per_token_above_272k_tokens_priority",
    "output_cost_per_token_above_272k_tokens_ultrafast",
    "output_cost_per_token_above_512k_tokens",
    "output_cost_per_token_balanced",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
    "output_cost_per_token_ultrafast",
    "output_cost_per_video",
    "output_cost_per_video_clip",
    "output_cost_per_video_per_second",
    "output_cost_per_video_token",
    "output_vector_size",
    "regional_endpoint_uplift_multiplier",
    "regional_processing_uplift_multiplier_eu",
    "regional_processing_uplift_multiplier_us",
    "search_context_cost_per_query",
    "search_queries_cost_per_query",
    "tiered_pricing",
];

/// Prices a call over the bundled catalog, with no discount or margin config.
pub(crate) struct CatalogPricing {
    catalog: Arc<Catalog>,
    context: PricingContext,
}

impl CatalogPricing {
    /// The catalog's own providers are the names a model path may start with.
    pub(crate) fn new(catalog: Arc<Catalog>) -> Self {
        let providers = catalog
            .served()
            .filter_map(|(_, fields)| fields.get("litellm_provider")?.as_str())
            .map(str::to_owned)
            .collect();
        let context = PricingContext {
            rules: catalog.capability_rules().clone(),
            names: KnownNames {
                providers,
                renamed_models: RENAMED_MODELS
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<HashSet<_>>(),
            },
            ..PricingContext::default()
        };
        Self { catalog, context }
    }
}

impl Pricing for CatalogPricing {
    fn price(&self, call: &PricedCall) -> Option<Receipt> {
        let lookup = CostLookup::new(&*self.catalog, &self.context);
        price_call(&lookup, call, jiff::Timestamp::now())
    }
}

/// What a call to `deployment` is priced with. `None`, so the call goes unpriced, for one Python
/// prices another way: custom pricing in its params or `model_info`, a `pricing_provider`, or a
/// Bedrock `model_id`. A Vertex AI or Bedrock Mantle deployment is unpriced too, since its price
/// reads a region this does not resolve yet.
pub(crate) fn identity(deployment: &Deployment) -> Option<PricingIdentity> {
    let params = &deployment.params;
    let model_info = &deployment.model_info.raw;
    let set = |value: Option<&Value>| value.is_some_and(|value| !value.is_null());
    let custom = CUSTOM_PRICING_KEYS.iter().any(|key| {
        (*key != "off_peak_pricing" && set(params.extra.get(*key))) || set(model_info.get(*key))
    });
    let provider = deployment
        .resolved_provider
        .as_deref()
        .or(params.custom_llm_provider.as_deref())
        .or_else(|| params.model.split_once('/').map(|(provider, _)| provider));
    let regional = match provider {
        Some(provider) if provider.starts_with("vertex_ai") => true,
        Some("bedrock_mantle") => true,
        Some("bedrock") => params
            .model
            .trim_start_matches("bedrock/")
            .starts_with("mantle/"),
        _ => false,
    };
    if custom
        || regional
        || set(params.extra.get("model_id"))
        || set(model_info.get("pricing_provider"))
    {
        return None;
    }
    let text = |value: Option<&Value>| value.and_then(Value::as_str).map(str::to_owned);
    Some(PricingIdentity {
        deployment_id: Some(deployment.id.clone()),
        base_model: text(params.extra.get("base_model"))
            .or_else(|| text(model_info.get("base_model"))),
        region: None,
        vertex_location: None,
        data_residency: text(params.extra.get("data_residency")),
    })
}

#[cfg(test)]
mod tests {
    use litellm_cost::query::{Modalities, TokenUsage};
    use litellm_model_catalog::BUNDLED_REGISTRY;
    use rstest::rstest;

    use super::*;

    /// The bundled catalog prices a Claude call at its row's own rates.
    #[rstest]
    fn a_call_is_priced_at_its_catalog_rows_rates() {
        let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
        let row = catalog.get("claude-opus-5-5").unwrap();
        let rate = |key: &str| row[key].as_f64().unwrap();
        let listed = 1000.0 * rate("input_cost_per_token") + 100.0 * rate("output_cost_per_token");
        let pricing = CatalogPricing::new(Arc::new(catalog));

        let receipt = pricing
            .price(&PricedCall {
                provider: "anthropic".to_owned(),
                request_model: "claude-opus-5-5".to_owned(),
                response_model: Some("claude-opus-5-5".to_owned()),
                identity: PricingIdentity {
                    deployment_id: Some("deployment-1".to_owned()),
                    ..PricingIdentity::default()
                },
                usage: TokenUsage {
                    prompt_tokens: 1000,
                    input: Modalities {
                        text: 1000,
                        ..Modalities::default()
                    },
                    output: Modalities {
                        text: 100,
                        ..Modalities::default()
                    },
                    ..TokenUsage::default()
                },
                service_tier: None,
            })
            .unwrap();

        assert!((receipt.total - listed).abs() < 1e-12);
    }
}
