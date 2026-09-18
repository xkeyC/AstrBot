<template>
  <div class="dashboard-page permission-rules-page" :class="{ 'is-dark': isDark }">
    <v-container fluid class="dashboard-shell pa-4 pa-md-6">
      <div class="dashboard-header">
        <div class="dashboard-header-main">
          <h1 class="dashboard-title">{{ tm('page.title') }}</h1>
          <p class="dashboard-subtitle">{{ tm('page.subtitle') }}</p>
        </div>

        <div class="dashboard-header-actions">
          <v-btn variant="text" color="primary" prepend-icon="mdi-refresh" :loading="loading" @click="reload">
            {{ tm('actions.refresh') }}
          </v-btn>
          <v-btn variant="tonal" color="primary" prepend-icon="mdi-content-save" :loading="saving" @click="save">
            {{ tm('actions.save') }}
          </v-btn>
        </div>
      </div>

      <div v-if="hasUnsavedChanges" class="unsaved-banner">
        <v-icon size="18" color="warning">mdi-alert-circle-outline</v-icon>
        <span>{{ tm('messages.unsavedChangesNotice') }}</span>
      </div>

      <!-- Rules -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('rules.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('rules.subtitle') }}</div>
        </div>
        <div class="dashboard-section-actions">
          <v-btn color="primary" variant="tonal" prepend-icon="mdi-plus" @click="addRule">
            {{ tm('rules.add') }}
          </v-btn>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5">
        <div class="syntax-help">
          <v-icon size="18" color="primary">mdi-information-outline</v-icon>
          <div>
            <div>{{ tm('match.syntaxTitle') }}</div>
            <ul class="syntax-list">
              <li><code>&lt;group_id&gt;/&lt;sender_id&gt;</code> — {{ tm('match.syntaxGroupSender') }}</li>
              <li><code>p_&lt;sender_id&gt;</code> — {{ tm('match.syntaxSender') }}</li>
              <li><code>g_&lt;group_id&gt;</code> — {{ tm('match.syntaxGroup') }}</li>
              <li><code>role:admin</code> / <code>role:member</code> — {{ tm('match.syntaxRole') }}</li>
              <li><code>*</code> — {{ tm('match.syntaxAll') }}</li>
            </ul>
          </div>
        </div>

        <div v-if="rules.length === 0" class="dashboard-empty mt-4">
          {{ tm('rules.empty') }}
        </div>

        <div v-else class="rule-list mt-4">
          <div
            v-for="(rule, idx) in rules"
            :key="rule.__key"
            class="rule-card"
            :class="{ 'rule-card--disabled': !rule.enabled, 'rule-card--hit': testResult?.index === idx }"
          >
            <div class="rule-head">
              <span class="rule-index">#{{ idx + 1 }}</span>
              <v-text-field
                v-model="rule.name"
                :label="tm('rules.name')"
                :placeholder="tm('rules.namePlaceholder')"
                variant="outlined"
                density="compact"
                hide-details
                class="rule-name"
              />
              <v-switch
                v-model="rule.enabled"
                :label="rule.enabled ? tm('rules.enabled') : tm('rules.disabled')"
                color="primary"
                density="compact"
                hide-details
                inset
                class="rule-switch"
              />
              <div class="rule-actions">
                <v-btn
                  icon="mdi-arrow-up"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :disabled="idx === 0"
                  :title="tm('rules.moveUp')"
                  @click="moveRule(idx, -1)"
                />
                <v-btn
                  icon="mdi-arrow-down"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :disabled="idx === rules.length - 1"
                  :title="tm('rules.moveDown')"
                  @click="moveRule(idx, 1)"
                />
                <v-btn
                  icon="mdi-content-copy"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :title="tm('rules.duplicate')"
                  @click="duplicateRule(idx)"
                />
                <v-btn
                  icon="mdi-delete-outline"
                  size="small"
                  variant="text"
                  color="error"
                  density="comfortable"
                  :title="tm('rules.delete')"
                  @click="deleteRule(idx)"
                />
                <v-btn
                  :icon="rule.__expanded ? 'mdi-chevron-up' : 'mdi-chevron-down'"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :title="rule.__expanded ? tm('rules.collapse') : tm('rules.expand')"
                  @click="rule.__expanded = !rule.__expanded"
                />
              </div>
            </div>

            <div v-if="!rule.__expanded" class="rule-summary">
              <v-chip
                v-for="cond in rule.match"
                :key="cond"
                size="small"
                label
                :color="conditionError(cond) ? 'error' : 'primary'"
                variant="tonal"
              >
                {{ cond }}
              </v-chip>
              <span v-if="rule.match.length === 0" class="setting-subtitle mt-0">{{ tm('match.none') }}</span>
              <span v-if="ruleSummary(rule)" class="setting-subtitle mt-0">· {{ ruleSummary(rule) }}</span>
            </div>

            <div v-else class="rule-body">
              <v-combobox
                :model-value="rule.match"
                :label="tm('match.label')"
                :hint="tm('match.hint')"
                :error-messages="matchErrors(rule)"
                :delimiters="[',', ' ']"
                persistent-hint
                multiple
                chips
                closable-chips
                variant="outlined"
                density="comfortable"
                @update:model-value="(v: unknown) => (rule.match = cleanList(v))"
              >
                <template #chip="{ props: chipProps, item }">
                  <v-chip
                    v-bind="chipProps"
                    :color="conditionError(String(item.raw)) ? 'error' : 'primary'"
                    size="small"
                    label
                  >
                    {{ item.raw }}
                  </v-chip>
                </template>
              </v-combobox>

              <div class="section-label">{{ tm('fields.toolsTitle') }}</div>
              <div class="dashboard-form-grid">
                <v-combobox
                  v-for="field in patternFields"
                  :key="field.key"
                  :model-value="rule[field.key]"
                  :items="field.kind === 'tool' ? toolNames : mcpServerNames"
                  :label="tm(field.label)"
                  :hint="tm(field.hint)"
                  :loading="field.kind === 'tool' ? toolsLoading : mcpLoading"
                  :delimiters="[',', ' ']"
                  persistent-hint
                  multiple
                  chips
                  closable-chips
                  variant="outlined"
                  density="comfortable"
                  @update:model-value="(v: unknown) => (rule[field.key] = cleanList(v))"
                >
                  <template #chip="{ props: chipProps, item }">
                    <v-chip
                      v-bind="chipProps"
                      :color="field.key.endsWith('_deny') ? 'error' : 'success'"
                      size="small"
                      label
                    >
                      {{ item.raw }}
                    </v-chip>
                  </template>
                </v-combobox>
              </div>

              <div class="section-label">{{ tm('fields.overridesTitle') }}</div>
              <div class="dashboard-form-grid">
                <v-select
                  v-model="rule.persona_id"
                  :items="personaItems(rule.persona_id)"
                  :label="tm('fields.persona')"
                  :hint="tm('fields.personaHint')"
                  :loading="personasLoading"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
                <v-combobox
                  :model-value="rule.model"
                  :items="modelIds"
                  :label="tm('fields.model')"
                  :hint="rule.model ? tm('fields.modelCacheWarning') : tm('fields.modelHint')"
                  :loading="modelsLoading"
                  :color="rule.model ? 'warning' : undefined"
                  persistent-hint
                  clearable
                  variant="outlined"
                  density="comfortable"
                  @update:model-value="(v: unknown) => (rule.model = str(v).trim())"
                >
                  <template #item="{ props: itemProps, item }">
                    <v-list-item
                      v-bind="itemProps"
                      :title="modelDisplayName(String(item.raw))"
                      :subtitle="String(item.raw)"
                    />
                  </template>
                </v-combobox>
                <v-select
                  v-model="rule.native_exec"
                  :items="triStateItems"
                  :label="tm('fields.nativeExec')"
                  :hint="tm('fields.nativeExecHint')"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
                <v-select
                  v-model="rule.global_memory"
                  :items="triStateItems"
                  :label="tm('fields.globalMemory')"
                  :hint="tm('fields.globalMemoryHint')"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
              </div>
              <div v-if="rule.model" class="model-warning">
                <v-icon size="16" color="warning">mdi-alert-outline</v-icon>
                <span>{{ tm('fields.modelCacheWarning') }}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Tester -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('tester.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('tester.subtitle') }}</div>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5">
        <div class="tester-grid">
          <v-text-field
            v-model="tester.sender_id"
            :label="tm('tester.senderId')"
            variant="outlined"
            density="comfortable"
            hide-details
          />
          <v-text-field
            v-model="tester.group_id"
            :label="tm('tester.groupId')"
            :placeholder="tm('tester.groupIdPlaceholder')"
            variant="outlined"
            density="comfortable"
            hide-details
          />
          <v-select
            v-model="tester.role"
            :items="roleItems"
            :label="tm('tester.role')"
            variant="outlined"
            density="comfortable"
            hide-details
          />
        </div>

        <div class="tester-result mt-4">
          <template v-if="testResult">
            <div class="tester-verdict">
              <v-icon color="success">mdi-check-circle</v-icon>
              <span>
                {{ tm('tester.matched', { index: testResult.index + 1, name: ruleLabel(rules[testResult.index], testResult.index) }) }}
              </span>
              <span class="setting-subtitle mt-0">({{ tm('tester.matchedBy', { condition: testResult.condition }) }})</span>
            </div>
            <div class="dashboard-meta-list tester-policy">
              <div v-for="line in policyLines(rules[testResult.index])" :key="line.label" class="tester-policy-row">
                <span class="tester-policy-label">{{ line.label }}</span>
                <span class="tester-policy-value">{{ line.value }}</span>
              </div>
            </div>
          </template>
          <div v-else class="tester-verdict">
            <v-icon color="info">mdi-information-outline</v-icon>
            <span>{{ tm('tester.noMatch') }}</span>
          </div>

          <div v-if="rules.length" class="tester-trace mt-3">
            <div v-for="(status, idx) in testTrace" :key="rules[idx].__key" class="tester-trace-row">
              <v-chip size="x-small" label variant="tonal" :color="traceColor(status)">
                {{ tm(`tester.status.${status}`) }}
              </v-chip>
              <span>#{{ idx + 1 }} {{ ruleLabel(rules[idx], idx) }}</span>
            </div>
          </div>
        </div>
      </section>

      <v-snackbar v-model="snackbar.show" :color="snackbar.color" timeout="3000" location="top">
        {{ snackbar.message }}
        <template #actions>
          <v-btn variant="text" @click="snackbar.show = false">{{ tm('actions.close') }}</v-btn>
        </template>
      </v-snackbar>
    </v-container>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import { useTheme } from 'vuetify'
import { httpClient } from '@/api/http'
import { configProfileApi, mcpApi, personaApi, toolApi } from '@/api/v1'
import { useModuleI18n } from '@/i18n/composables'
import { askForConfirmation, useConfirmDialog } from '@/utils/confirmDialog'

type TriState = 'inherit' | 'allow' | 'deny'
type PatternKey = 'tools_allow' | 'tools_deny' | 'mcp_allow' | 'mcp_deny'

type RuleRow = {
  __key: string
  __expanded: boolean
  // Keys this page does not know about, preserved on save.
  __extra: Record<string, unknown>
  name: string
  enabled: boolean
  match: string[]
  tools_allow: string[]
  tools_deny: string[]
  mcp_allow: string[]
  mcp_deny: string[]
  persona_id: string
  model: string
  native_exec: TriState
  global_memory: TriState
}

type Facts = { sender_id: string; group_id: string; role: string }
type TraceStatus = 'matched' | 'shadowed' | 'noMatch' | 'disabled'

const CONFIG_ID = 'default'
const CONFIG_KEY = 'permission_rules'
const KNOWN_KEYS = [
  'name',
  'enabled',
  'match',
  'tools_allow',
  'tools_deny',
  'mcp_allow',
  'mcp_deny',
  'persona_id',
  'model',
  'native_exec',
  'global_memory'
]

const { tm } = useModuleI18n('features/permissions')
const theme = useTheme()
const confirmDialog = useConfirmDialog()
const isDark = computed(() => theme.global.current.value.dark)

const loading = ref(false)
const saving = ref(false)
const hasLoaded = ref(false)
const initialSnapshot = ref('')
const rules = ref<RuleRow[]>([])
const snackbar = ref({ show: false, message: '', color: 'success' })

const toolNames = ref<string[]>([])
const toolsLoading = ref(false)
const mcpServerNames = ref<string[]>([])
const mcpLoading = ref(false)
const personaIds = ref<string[]>([])
const personasLoading = ref(false)
const models = ref<{ model?: string; id?: string; display_name?: string }[]>([])
const modelsLoading = ref(false)

const tester = ref<Facts>({ sender_id: '', group_id: '', role: 'member' })

const patternFields: { key: PatternKey; kind: 'tool' | 'mcp'; label: string; hint: string }[] = [
  { key: 'tools_allow', kind: 'tool', label: 'fields.toolsAllow', hint: 'fields.toolsAllowHint' },
  { key: 'tools_deny', kind: 'tool', label: 'fields.toolsDeny', hint: 'fields.toolsDenyHint' },
  { key: 'mcp_allow', kind: 'mcp', label: 'fields.mcpAllow', hint: 'fields.mcpAllowHint' },
  { key: 'mcp_deny', kind: 'mcp', label: 'fields.mcpDeny', hint: 'fields.mcpDenyHint' }
]

const triStateItems = computed(() => [
  { title: tm('triState.inherit'), value: 'inherit' },
  { title: tm('triState.allow'), value: 'allow' },
  { title: tm('triState.deny'), value: 'deny' }
])

const roleItems = computed(() => [
  { title: tm('tester.roleMember'), value: 'member' },
  { title: tm('tester.roleAdmin'), value: 'admin' }
])

function toast(message: string, color: 'success' | 'error' | 'warning' = 'success') {
  snackbar.value = { show: true, message, color }
}

function errorMessage(e: any, fallback: string): string {
  if (typeof e === 'string') return e
  return e?.response?.data?.message || e?.message || fallback
}

function newKey(): string {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : value == null ? '' : String(value)
}

// ---------- value coercion (mirrors astrbot/core/permission_rules.py) ----------

/** `_as_list`: a string is split on commas/newlines; a list is stringified. */
function asList(value: unknown): string[] {
  let items: string[]
  if (typeof value === 'string') items = value.replace(/,/g, '\n').split(/\r\n|\r|\n/)
  else if (Array.isArray(value)) items = value.map((v) => String(v))
  else items = []
  return items.map((i) => i.trim()).filter((i) => !!i)
}

/** `_as_opt_bool`: booleans, or the strings "true"/"false"; anything else is null. */
function asOptBool(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value
  if (typeof value === 'string' && ['true', 'false'].includes(value.toLowerCase())) {
    return value.toLowerCase() === 'true'
  }
  return null
}

function toTriState(value: unknown): TriState {
  const b = asOptBool(value)
  return b === null ? 'inherit' : b ? 'allow' : 'deny'
}

function fromTriState(value: TriState): boolean | null {
  return value === 'allow' ? true : value === 'deny' ? false : null
}

/** Combobox output → trimmed, de-duplicated list of non-empty strings. */
function cleanList(value: unknown): string[] {
  const list = Array.isArray(value) ? value : value == null ? [] : [value]
  const out: string[] = []
  for (const v of list) {
    const s = str(typeof v === 'object' && v !== null && 'value' in v ? (v as any).value : v).trim()
    if (s && !out.includes(s)) out.push(s)
  }
  return out
}

// ---------- matching (mirrors condition_matches / resolve_policy) ----------

function conditionMatches(condition: string, facts: Facts): boolean {
  const cond = condition.trim()
  if (!cond) return false
  if (cond === '*') return true
  if (cond.startsWith('role:')) return facts.role === cond.slice('role:'.length)
  if (cond.startsWith('p_')) return facts.sender_id === cond.slice(2)
  if (cond.startsWith('g_')) return !!facts.group_id && facts.group_id === cond.slice(2)
  if (cond.includes('/')) {
    const i = cond.indexOf('/')
    return facts.group_id === cond.slice(0, i) && facts.sender_id === cond.slice(i + 1)
  }
  return false
}

/** Returns an error message when the condition can never match, else ''. */
function conditionError(condition: string): string {
  const cond = condition.trim()
  if (!cond || cond === '*') return ''
  if (cond.startsWith('role:')) {
    const role = cond.slice('role:'.length)
    return role === 'admin' || role === 'member' ? '' : tm('match.errorRole', { condition: cond })
  }
  if (cond.startsWith('p_') || cond.startsWith('g_')) {
    return cond.length > 2 ? '' : tm('match.errorEmptyId', { condition: cond })
  }
  if (cond.includes('/')) {
    const i = cond.indexOf('/')
    return cond.slice(0, i) && cond.slice(i + 1) ? '' : tm('match.errorEmptyId', { condition: cond })
  }
  return tm('match.errorSyntax', { condition: cond })
}

function matchErrors(rule: RuleRow): string[] {
  const errs = rule.match.map(conditionError).filter((e) => !!e)
  if (rule.enabled && rule.match.length === 0) errs.unshift(tm('match.errorNone'))
  return errs
}

const normalizedFacts = computed<Facts>(() => ({
  sender_id: tester.value.sender_id.trim(),
  group_id: tester.value.group_id.trim(),
  role: tester.value.role || 'member'
}))

const testTrace = computed<TraceStatus[]>(() => {
  let found = false
  return rules.value.map((rule) => {
    if (!rule.enabled) return 'disabled'
    const hit = rule.match.some((c) => conditionMatches(c, normalizedFacts.value))
    if (!hit) return 'noMatch'
    if (found) return 'shadowed'
    found = true
    return 'matched'
  })
})

const testResult = computed(() => {
  const index = testTrace.value.indexOf('matched')
  if (index < 0) return null
  const condition = rules.value[index].match.find((c) => conditionMatches(c, normalizedFacts.value)) || ''
  return { index, condition }
})

function traceColor(status: TraceStatus): string {
  if (status === 'matched') return 'success'
  if (status === 'shadowed') return 'warning'
  return 'grey'
}

function ruleLabel(rule: RuleRow | undefined, idx: number): string {
  return rule?.name.trim() || tm('rules.unnamed', { index: idx + 1 })
}

function triLabel(value: TriState): string {
  return tm(`triState.${value}`)
}

function policyLines(rule: RuleRow | undefined): { label: string; value: string }[] {
  if (!rule) return []
  const list = (v: string[]) => (v.length ? v.join(', ') : '—')
  return [
    { label: tm('fields.toolsAllow'), value: list(rule.tools_allow) },
    { label: tm('fields.toolsDeny'), value: list(rule.tools_deny) },
    { label: tm('fields.mcpAllow'), value: list(rule.mcp_allow) },
    { label: tm('fields.mcpDeny'), value: list(rule.mcp_deny) },
    { label: tm('fields.persona'), value: rule.persona_id || tm('fields.noOverride') },
    { label: tm('fields.model'), value: rule.model || tm('fields.noOverride') },
    { label: tm('fields.nativeExec'), value: triLabel(rule.native_exec) },
    { label: tm('fields.globalMemory'), value: triLabel(rule.global_memory) }
  ]
}

function ruleSummary(rule: RuleRow): string {
  const parts: string[] = []
  if (rule.tools_allow.length || rule.tools_deny.length) parts.push(tm('fields.toolsTitle'))
  if (rule.mcp_allow.length || rule.mcp_deny.length) parts.push('MCP')
  if (rule.persona_id) parts.push(`${tm('fields.persona')}: ${rule.persona_id}`)
  if (rule.model) parts.push(`${tm('fields.model')}: ${rule.model}`)
  if (rule.native_exec !== 'inherit') parts.push(`${tm('fields.nativeExec')}: ${triLabel(rule.native_exec)}`)
  if (rule.global_memory !== 'inherit') parts.push(`${tm('fields.globalMemory')}: ${triLabel(rule.global_memory)}`)
  return parts.join(' · ')
}

// ---------- rule list editing ----------

function emptyRule(): RuleRow {
  return {
    __key: newKey(),
    __expanded: true,
    __extra: {},
    name: '',
    enabled: true,
    match: [],
    tools_allow: [],
    tools_deny: [],
    mcp_allow: [],
    mcp_deny: [],
    persona_id: '',
    model: '',
    native_exec: 'inherit',
    global_memory: 'inherit'
  }
}

function addRule() {
  rules.value.push(emptyRule())
}

function moveRule(idx: number, delta: number) {
  const target = idx + delta
  if (target < 0 || target >= rules.value.length) return
  const [row] = rules.value.splice(idx, 1)
  rules.value.splice(target, 0, row)
}

function duplicateRule(idx: number) {
  const src = rules.value[idx]
  const copy: RuleRow = {
    ...JSON.parse(JSON.stringify(src)),
    __key: newKey(),
    __expanded: true,
    name: src.name ? tm('rules.copyName', { name: src.name }) : ''
  }
  rules.value.splice(idx + 1, 0, copy)
}

async function deleteRule(idx: number) {
  const confirmed = await askForConfirmation(
    tm('messages.deleteConfirm', { name: ruleLabel(rules.value[idx], idx) }),
    confirmDialog
  )
  if (!confirmed) return
  rules.value.splice(idx, 1)
}

// ---------- suggestions ----------

async function loadTools() {
  toolsLoading.value = true
  try {
    const res = await toolApi.list()
    const data = res.data?.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
    toolNames.value = Array.from(new Set(data.map((t: any) => str(t?.name)).filter((n: string) => !!n))).sort()
  } catch {
    toolNames.value = []
  } finally {
    toolsLoading.value = false
  }
}

async function loadMcpServers() {
  mcpLoading.value = true
  try {
    const res = await mcpApi.list()
    const data = res.data?.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
    mcpServerNames.value = Array.from(new Set(data.map((s: any) => str(s?.name)).filter((n: string) => !!n))).sort()
  } catch {
    mcpServerNames.value = []
  } finally {
    mcpLoading.value = false
  }
}

async function loadPersonas() {
  personasLoading.value = true
  try {
    const res = await personaApi.list()
    const data = res.data?.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
    personaIds.value = Array.from(new Set(data.map((p: any) => str(p?.persona_id)).filter((n: string) => !!n)))
  } catch {
    personaIds.value = []
  } finally {
    personasLoading.value = false
  }
}

async function loadModels() {
  modelsLoading.value = true
  try {
    const res = await httpClient.get('/api/codex/models', { params: { include_hidden: false } })
    models.value = res.data.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
  } catch {
    models.value = []
  } finally {
    modelsLoading.value = false
  }
}

const modelIds = computed(() => models.value.map((m) => str(m.model || m.id)).filter((id) => !!id))

function modelDisplayName(id: string): string {
  const info = models.value.find((m) => (m.model || m.id) === id)
  return info?.display_name || id
}

function personaItems(current: string) {
  const items = [{ title: tm('fields.noOverride'), value: '' }]
  for (const id of personaIds.value) items.push({ title: id, value: id })
  if (current && !personaIds.value.includes(current)) {
    items.push({ title: tm('fields.personaMissing', { id: current }), value: current })
  }
  return items
}

// ---------- config load/save ----------

function normalizeRule(raw: Record<string, unknown>): RuleRow {
  const extra: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(raw)) {
    if (!KNOWN_KEYS.includes(k)) extra[k] = v
  }
  return {
    __key: newKey(),
    __expanded: false,
    __extra: extra,
    name: str(raw.name),
    // Only an explicit false (or "false") disables a rule, like the backend.
    enabled: asOptBool(raw.enabled === undefined ? true : raw.enabled) !== false,
    match: asList(raw.match),
    tools_allow: asList(raw.tools_allow),
    tools_deny: asList(raw.tools_deny),
    mcp_allow: asList(raw.mcp_allow),
    mcp_deny: asList(raw.mcp_deny),
    persona_id: str(raw.persona_id),
    model: str(raw.model),
    native_exec: toTriState(raw.native_exec),
    global_memory: toTriState(raw.global_memory)
  }
}

function normalizeRules(value: unknown): RuleRow[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((r): r is Record<string, unknown> => !!r && typeof r === 'object' && !Array.isArray(r))
    .map(normalizeRule)
}

function rulesPayload(rows: RuleRow[]) {
  return rows.map((r) => ({
    ...r.__extra,
    name: r.name.trim(),
    enabled: !!r.enabled,
    match: cleanList(r.match),
    tools_allow: cleanList(r.tools_allow),
    tools_deny: cleanList(r.tools_deny),
    mcp_allow: cleanList(r.mcp_allow),
    mcp_deny: cleanList(r.mcp_deny),
    persona_id: r.persona_id || '',
    model: (r.model || '').trim(),
    native_exec: fromTriState(r.native_exec),
    global_memory: fromTriState(r.global_memory)
  }))
}

const hasUnsavedChanges = computed(
  () => hasLoaded.value && JSON.stringify(rulesPayload(rules.value)) !== initialSnapshot.value
)

async function fetchDefaultConfig(): Promise<any> {
  const res = await configProfileApi.get(CONFIG_ID)
  if (res.data.status !== 'ok') {
    throw new Error(res.data.message || tm('messages.loadConfigFailed'))
  }
  return (res.data.data as any)?.config || {}
}

async function loadConfig() {
  try {
    const config = await fetchDefaultConfig()
    rules.value = normalizeRules(config?.[CONFIG_KEY])
    initialSnapshot.value = JSON.stringify(rulesPayload(rules.value))
    hasLoaded.value = true
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.loadConfigFailed')), 'error')
  }
}

async function confirmProblems(): Promise<boolean> {
  const problems: string[] = []
  rules.value.forEach((rule, idx) => {
    if (!rule.enabled) return
    const errs = matchErrors(rule)
    if (errs.length) problems.push(`#${idx + 1} ${ruleLabel(rule, idx)}: ${errs.join('; ')}`)
  })
  if (!problems.length) return true
  return askForConfirmation(
    `${tm('messages.invalidConditionsConfirm')}\n\n${problems.join('\n')}`,
    confirmDialog
  )
}

async function save() {
  if (!(await confirmProblems())) return
  saving.value = true
  try {
    // Re-fetch so that only permission_rules is replaced and every other
    // config field keeps its latest stored value.
    const config = await fetchDefaultConfig()
    const payload = rulesPayload(rules.value)
    config[CONFIG_KEY] = payload
    const res = await configProfileApi.update(CONFIG_ID, config)
    if (res.data.status === 'ok') {
      initialSnapshot.value = JSON.stringify(payload)
      toast(res.data.message || tm('messages.saveSuccess'))
    } else {
      toast(res.data.message || tm('messages.saveFailed'), 'error')
    }
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.saveFailed')), 'error')
  } finally {
    saving.value = false
  }
}

async function reload() {
  if (hasUnsavedChanges.value) {
    const confirmed = await askForConfirmation(tm('messages.unsavedChangesReloadConfirm'), confirmDialog)
    if (!confirmed) return
  }
  loading.value = true
  try {
    await Promise.all([loadConfig(), loadTools(), loadMcpServers(), loadPersonas(), loadModels()])
  } finally {
    loading.value = false
  }
}

function handleBeforeUnload(event: BeforeUnloadEvent) {
  if (!hasUnsavedChanges.value) return
  event.preventDefault()
  event.returnValue = ''
}

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload)
  reload()
})

onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload)
})

onBeforeRouteLeave(async () => {
  if (!hasUnsavedChanges.value) return true
  return askForConfirmation(tm('messages.unsavedChangesLeaveConfirm'), confirmDialog)
})
</script>

<style scoped>
@import '@/styles/dashboard-shell.css';

.permission-rules-page {
  padding-bottom: 40px;
}

.unsaved-banner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 14px;
  margin-bottom: 18px;
  border: 1px solid rgba(var(--v-theme-warning), 0.22);
  border-radius: 12px;
  background: rgba(var(--v-theme-warning), 0.08);
  color: var(--dashboard-text);
  font-size: 13px;
  line-height: 1.5;
}

.syntax-help {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  color: var(--dashboard-muted);
  font-size: 13px;
  line-height: 1.6;
}

.syntax-list {
  margin: 4px 0 0 18px;
}

.syntax-list code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--dashboard-text);
}

.rule-list {
  display: grid;
  gap: 14px;
}

.rule-card {
  border: 1px solid var(--dashboard-border);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(var(--v-theme-primary), 0.02);
  min-width: 0;
}

.rule-card--disabled {
  opacity: 0.65;
}

.rule-card--hit {
  border-color: rgba(var(--v-theme-success), 0.6);
}

.rule-head {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.rule-index {
  font-weight: 700;
  color: var(--dashboard-muted);
  min-width: 28px;
}

.rule-name {
  flex: 1 1 240px;
  min-width: 180px;
}

.rule-switch {
  flex: 0 0 auto;
}

.rule-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}

.rule-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}

.rule-body {
  margin-top: 14px;
}

.section-label {
  margin: 18px 0 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--dashboard-muted);
}

.setting-subtitle {
  margin-top: 6px;
  color: var(--dashboard-muted);
  font-size: 13px;
  line-height: 1.6;
}


.model-warning {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  font-size: 13px;
  color: rgb(var(--v-theme-warning));
}

.tester-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 200px;
  gap: 12px;
}

.tester-verdict {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  font-weight: 600;
}

.tester-policy {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 18px;
  margin-top: 10px;
}

.tester-policy-row {
  display: flex;
  gap: 8px;
  font-size: 13px;
  min-width: 0;
}

.tester-policy-label {
  color: var(--dashboard-muted);
  min-width: 140px;
}

.tester-policy-value {
  word-break: break-all;
}

.tester-trace {
  display: grid;
  gap: 4px;
  font-size: 13px;
}

.tester-trace-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

@media (max-width: 900px) {
  .tester-grid,
  .tester-policy {
    grid-template-columns: 1fr;
  }
}
</style>
