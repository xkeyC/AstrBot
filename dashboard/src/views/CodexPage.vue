<template>
  <div class="dashboard-page codex-page" :class="{ 'is-dark': isDark }">
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

      <!-- Account -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('account.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('account.subtitle') }}</div>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5 codex-section">
        <div class="account-status">
          <v-progress-circular v-if="accountLoading" indeterminate size="22" width="2" color="primary" />
          <template v-else-if="account && account.logged_in">
            <v-icon color="success">mdi-check-circle</v-icon>
            <div class="account-status-main">
              <div class="setting-title">{{ tm('account.loggedIn') }}</div>
              <div class="account-meta">
                <v-chip size="small" variant="tonal" color="primary" label>
                  {{ modeLabel(account.mode) }}
                </v-chip>
                <span v-if="account.email">{{ account.email }}</span>
                <v-chip v-if="account.plan" size="small" variant="tonal" label>{{ account.plan }}</v-chip>
                <span v-if="account.account_id" class="text-mono">{{ account.account_id }}</span>
              </div>
            </div>
            <v-btn
              variant="tonal"
              color="error"
              prepend-icon="mdi-logout"
              :loading="loggingOut"
              @click="logout"
            >
              {{ tm('account.logout') }}
            </v-btn>
          </template>
          <template v-else>
            <v-icon color="warning">mdi-account-alert-outline</v-icon>
            <div class="account-status-main">
              <div class="setting-title">{{ tm('account.notLoggedIn') }}</div>
              <div class="setting-subtitle">{{ accountError || tm('account.notLoggedInHint') }}</div>
            </div>
          </template>
        </div>

        <v-divider class="my-5" />

        <div class="login-grid">
          <div class="setting-card">
            <div class="setting-title">{{ tm('account.apiKeyTitle') }}</div>
            <div class="setting-subtitle mb-4">{{ tm('account.apiKeyHint') }}</div>
            <v-text-field
              v-model="apiKeyInput"
              :label="tm('account.apiKeyLabel')"
              type="password"
              autocomplete="off"
              variant="outlined"
              density="comfortable"
              hide-details="auto"
              @keyup.enter="loginWithApiKey"
            />
            <v-btn
              class="mt-4"
              color="primary"
              variant="tonal"
              prepend-icon="mdi-key-variant"
              :loading="apiKeyLoading"
              :disabled="!apiKeyInput.trim()"
              @click="loginWithApiKey"
            >
              {{ tm('account.apiKeyLogin') }}
            </v-btn>
          </div>

          <div class="setting-card">
            <div class="setting-title">{{ tm('account.chatgptTitle') }}</div>
            <div class="setting-subtitle mb-4">{{ tm('account.chatgptHint') }}</div>

            <template v-if="device">
              <div class="device-box">
                <div class="device-row">
                  <span class="device-label">{{ tm('account.deviceUrl') }}</span>
                  <a :href="device.verification_url" target="_blank" rel="noopener noreferrer">
                    {{ device.verification_url }}
                  </a>
                </div>
                <div class="device-row">
                  <span class="device-label">{{ tm('account.deviceCode') }}</span>
                  <span class="device-code">{{ device.user_code }}</span>
                  <v-btn
                    icon="mdi-content-copy"
                    size="small"
                    variant="text"
                    density="comfortable"
                    @click="copyCode"
                  />
                </div>
                <div class="device-row device-status">
                  <v-progress-circular
                    v-if="device.status === 'pending'"
                    indeterminate
                    size="16"
                    width="2"
                    color="primary"
                  />
                  <v-icon v-else-if="device.status === 'success'" size="18" color="success">mdi-check-circle</v-icon>
                  <v-icon v-else size="18" color="error">mdi-close-circle</v-icon>
                  <span>{{ deviceStatusText }}</span>
                </div>
              </div>
              <div class="d-flex flex-wrap mt-4" style="gap: 8px;">
                <v-btn
                  v-if="device.status === 'pending'"
                  variant="text"
                  color="error"
                  prepend-icon="mdi-cancel"
                  :loading="deviceCancelling"
                  @click="cancelDeviceLogin"
                >
                  {{ tm('account.deviceCancel') }}
                </v-btn>
                <v-btn
                  v-else
                  variant="tonal"
                  color="primary"
                  prepend-icon="mdi-restart"
                  :loading="deviceStarting"
                  @click="startDeviceLogin"
                >
                  {{ tm('account.deviceRestart') }}
                </v-btn>
              </div>
            </template>
            <v-btn
              v-else
              color="primary"
              variant="tonal"
              prepend-icon="mdi-open-in-new"
              :loading="deviceStarting"
              @click="startDeviceLogin"
            >
              {{ tm('account.chatgptLogin') }}
            </v-btn>
          </div>
        </div>
      </section>

      <!-- Model -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('model.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('model.subtitle') }}</div>
        </div>
        <div class="dashboard-section-actions">
          <v-btn
            variant="text"
            color="primary"
            prepend-icon="mdi-refresh"
            :loading="modelsLoading"
            @click="loadModels"
          >
            {{ tm('model.refreshModels') }}
          </v-btn>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5 codex-section">
        <div class="dashboard-form-grid">
          <v-combobox
            v-model="form.model"
            :items="modelIds"
            :label="tm('model.modelLabel')"
            :hint="modelsError || tm('model.modelHint')"
            :loading="modelsLoading"
            persistent-hint
            clearable
            variant="outlined"
            density="comfortable"
          >
            <template #item="{ props: itemProps, item }">
              <v-list-item
                v-bind="itemProps"
                :title="modelInfo(item.raw)?.display_name || item.raw"
                :subtitle="modelSubtitle(item.raw)"
              />
            </template>
          </v-combobox>

          <v-select
            v-model="form.reasoning_effort"
            :items="reasoningEffortItems"
            :label="tm('model.reasoningLabel')"
            :hint="tm('model.reasoningHint')"
            persistent-hint
            variant="outlined"
            density="comfortable"
          />

          <v-select
            v-model="form.model_provider"
            :items="providerItems"
            :label="tm('model.providerLabel')"
            :hint="tm('model.providerHint')"
            persistent-hint
            variant="outlined"
            density="comfortable"
          />
        </div>
        <div v-if="selectedModel?.description" class="setting-subtitle mt-4">
          {{ selectedModel.description }}
        </div>
      </section>

      <!-- Custom providers -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('providers.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('providers.subtitle') }}</div>
        </div>
        <div class="dashboard-section-actions">
          <v-btn color="primary" variant="tonal" prepend-icon="mdi-plus" @click="addProvider">
            {{ tm('providers.add') }}
          </v-btn>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5 codex-section">
        <div v-if="form.model_providers.length === 0" class="dashboard-empty">
          {{ tm('providers.empty') }}
        </div>
        <div v-else class="provider-list">
          <div v-for="(p, idx) in form.model_providers" :key="p.__key" class="provider-row">
            <v-text-field
              v-model="p.id"
              :label="tm('providers.id')"
              :rules="[providerIdRule]"
              variant="outlined"
              density="comfortable"
              hide-details="auto"
            />
            <v-text-field
              v-model="p.name"
              :label="tm('providers.name')"
              variant="outlined"
              density="comfortable"
              hide-details="auto"
            />
            <v-text-field
              v-model="p.base_url"
              :label="tm('providers.baseUrl')"
              :rules="[(v: string) => !!(v || '').trim() || tm('messages.baseUrlRequired')]"
              placeholder="https://api.example.com/v1"
              variant="outlined"
              density="comfortable"
              hide-details="auto"
            />
            <v-text-field
              v-model="p.api_key"
              :label="tm('providers.apiKey')"
              :type="p.__showKey ? 'text' : 'password'"
              :append-inner-icon="p.__showKey ? 'mdi-eye-off' : 'mdi-eye'"
              autocomplete="off"
              variant="outlined"
              density="comfortable"
              hide-details="auto"
              @click:append-inner="p.__showKey = !p.__showKey"
            />
            <v-select
              v-model="p.wire_api"
              :items="wireApiItems"
              :label="tm('providers.wireApi')"
              variant="outlined"
              density="comfortable"
              hide-details="auto"
            />
            <v-btn
              icon="mdi-delete-outline"
              variant="text"
              color="error"
              density="comfortable"
              class="provider-remove"
              @click="removeProvider(idx)"
            />
          </div>
        </div>
      </section>

      <!-- System prompt -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('prompt.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('prompt.subtitle') }}</div>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5 codex-section">
        <div class="dashboard-form-grid dashboard-form-grid--single">
          <v-textarea
            v-model="form.base_instructions"
            :label="tm('prompt.baseLabel')"
            :hint="tm('prompt.baseHint')"
            persistent-hint
            variant="outlined"
            density="comfortable"
            rows="6"
            auto-grow
          />
          <v-textarea
            v-model="form.developer_instructions"
            :label="tm('prompt.developerLabel')"
            :hint="tm('prompt.developerHint')"
            persistent-hint
            variant="outlined"
            density="comfortable"
            rows="4"
            auto-grow
          />
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
import { configProfileApi } from '@/api/v1'
import { useModuleI18n } from '@/i18n/composables'
import { askForConfirmation, useConfirmDialog } from '@/utils/confirmDialog'

type CodexAccount = {
  logged_in: boolean
  mode?: string
  email?: string | null
  account_id?: string | null
  plan?: string | null
  model?: string
  model_provider?: string
}

type CodexModel = {
  id?: string
  model?: string
  display_name?: string
  description?: string
  is_default?: boolean
  default_reasoning_effort?: string
  reasoning_efforts?: EffortEntry[]
}

type EffortEntry = string | { reasoning_effort?: string; effort?: string; description?: string }

type DeviceLogin = {
  login_id: string
  verification_url: string
  user_code: string
  status: 'pending' | 'success' | 'failed' | 'unknown'
  error?: string
}

type ProviderRow = {
  __key: string
  __showKey: boolean
  id: string
  name: string
  base_url: string
  api_key: string
  wire_api: string
}

type CodexForm = {
  model: string
  model_provider: string
  reasoning_effort: string
  model_providers: ProviderRow[]
  base_instructions: string
  developer_instructions: string
}

const CONFIG_ID = 'default'
const DEVICE_POLL_INTERVAL = 3000
const DEFAULT_EFFORTS = ['minimal', 'low', 'medium', 'high']
const PROVIDER_ID_RE = /^[A-Za-z0-9_-]+$/

const { tm } = useModuleI18n('features/codex')
const theme = useTheme()
const confirmDialog = useConfirmDialog()
const isDark = computed(() => theme.global.current.value.dark)

const loading = ref(false)
const saving = ref(false)
const hasLoaded = ref(false)
const initialSnapshot = ref('')

const account = ref<CodexAccount | null>(null)
const accountLoading = ref(false)
const accountError = ref('')
const loggingOut = ref(false)
const apiKeyInput = ref('')
const apiKeyLoading = ref(false)

const device = ref<DeviceLogin | null>(null)
const deviceStarting = ref(false)
const deviceCancelling = ref(false)
let devicePollTimer: ReturnType<typeof setInterval> | null = null
let devicePolling = false

const models = ref<CodexModel[]>([])
const modelsLoading = ref(false)
const modelsError = ref('')

const snackbar = ref({ show: false, message: '', color: 'success' })

const form = ref<CodexForm>(emptyForm())

// Codex only speaks the Responses API.
const wireApiItems = ['responses']

function emptyForm(): CodexForm {
  return {
    model: '',
    model_provider: '',
    reasoning_effort: '',
    model_providers: [],
    base_instructions: '',
    developer_instructions: ''
  }
}

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

// ---------- account ----------

function modeLabel(mode?: string): string {
  if (mode === 'apikey') return tm('account.modeApiKey')
  if (mode === 'chatgpt') return tm('account.modeChatgpt')
  return mode || tm('account.modeUnknown')
}

async function loadAccount() {
  accountLoading.value = true
  accountError.value = ''
  try {
    const res = await httpClient.get('/api/codex/account')
    if (res.data.status === 'ok') {
      account.value = res.data.data as CodexAccount
    } else {
      account.value = null
      accountError.value = res.data.message || tm('messages.accountLoadFailed')
    }
  } catch (e: any) {
    account.value = null
    accountError.value = errorMessage(e, tm('messages.accountLoadFailed'))
  } finally {
    accountLoading.value = false
  }
}

async function loginWithApiKey() {
  const key = apiKeyInput.value.trim()
  if (!key) return
  apiKeyLoading.value = true
  try {
    const res = await httpClient.post('/api/codex/login/api-key', { api_key: key })
    if (res.data.status === 'ok') {
      apiKeyInput.value = ''
      account.value = res.data.data as CodexAccount
      toast(tm('messages.loginSuccess'))
      loadModels()
    } else {
      toast(res.data.message || tm('messages.loginFailed'), 'error')
    }
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.loginFailed')), 'error')
  } finally {
    apiKeyLoading.value = false
  }
}

async function logout() {
  const confirmed = await askForConfirmation(tm('messages.logoutConfirm'), confirmDialog)
  if (!confirmed) return
  loggingOut.value = true
  try {
    const res = await httpClient.post('/api/codex/logout')
    if (res.data.status === 'ok') {
      toast(tm('messages.logoutSuccess'))
    } else {
      toast(res.data.message || tm('messages.logoutFailed'), 'error')
    }
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.logoutFailed')), 'error')
  } finally {
    loggingOut.value = false
    await loadAccount()
  }
}

// ---------- device-code (ChatGPT) login ----------

const deviceStatusText = computed(() => {
  const d = device.value
  if (!d) return ''
  if (d.status === 'pending') return tm('account.devicePending')
  if (d.status === 'success') return tm('account.deviceSuccess')
  return d.error || tm('account.deviceFailed')
})

function stopDevicePolling() {
  if (devicePollTimer !== null) {
    clearInterval(devicePollTimer)
    devicePollTimer = null
  }
}

async function startDeviceLogin() {
  stopDevicePolling()
  deviceStarting.value = true
  try {
    const res = await httpClient.post('/api/codex/login/device')
    if (res.data.status !== 'ok') {
      toast(res.data.message || tm('messages.deviceStartFailed'), 'error')
      return
    }
    const data = res.data.data || {}
    device.value = {
      login_id: str(data.login_id),
      verification_url: str(data.verification_url),
      user_code: str(data.user_code),
      status: 'pending'
    }
    devicePollTimer = setInterval(pollDeviceLogin, DEVICE_POLL_INTERVAL)
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.deviceStartFailed')), 'error')
  } finally {
    deviceStarting.value = false
  }
}

async function pollDeviceLogin() {
  const d = device.value
  if (!d || d.status !== 'pending') {
    stopDevicePolling()
    return
  }
  if (devicePolling) return
  devicePolling = true
  try {
    const res = await httpClient.get(`/api/codex/login/device/${encodeURIComponent(d.login_id)}`)
    // Ignore results for a flow that was cancelled/restarted meanwhile.
    if (device.value?.login_id !== d.login_id) return
    if (res.data.status !== 'ok') {
      d.status = 'failed'
      d.error = res.data.message
    } else {
      const status = str(res.data.data?.status) || 'unknown'
      if (status === 'pending') return
      d.status = status === 'success' ? 'success' : status === 'failed' ? 'failed' : 'unknown'
      d.error = res.data.data?.error || undefined
    }
    stopDevicePolling()
    if (d.status === 'success') {
      toast(tm('messages.loginSuccess'))
      await loadAccount()
      loadModels()
    } else {
      toast(d.error || tm('account.deviceFailed'), 'error')
    }
  } catch (e: any) {
    if (device.value?.login_id !== d.login_id) return
    d.status = 'failed'
    d.error = errorMessage(e, tm('account.deviceFailed'))
    stopDevicePolling()
  } finally {
    devicePolling = false
  }
}

async function cancelDeviceLogin() {
  const d = device.value
  if (!d) return
  stopDevicePolling()
  deviceCancelling.value = true
  try {
    await httpClient.delete(`/api/codex/login/device/${encodeURIComponent(d.login_id)}`)
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.deviceCancelFailed')), 'error')
  } finally {
    deviceCancelling.value = false
    device.value = null
  }
}

async function copyCode() {
  if (!device.value?.user_code) return
  try {
    await navigator.clipboard.writeText(device.value.user_code)
    toast(tm('messages.copied'))
  } catch {
    toast(tm('messages.copyFailed'), 'warning')
  }
}

// ---------- models ----------

const modelIds = computed(() =>
  models.value.map((m) => str(m.model || m.id)).filter((id) => !!id)
)

function modelInfo(id: string): CodexModel | undefined {
  return models.value.find((m) => (m.model || m.id) === id)
}

function modelSubtitle(id: string): string {
  const info = modelInfo(id)
  const parts = [id]
  if (info?.is_default) parts.push(tm('model.defaultTag'))
  return parts.join(' · ')
}

const selectedModel = computed(() => {
  if (form.value.model) return modelInfo(form.value.model)
  return models.value.find((m) => m.is_default)
})

function effortValue(e: EffortEntry): string {
  if (typeof e === 'string') return e
  return str(e?.reasoning_effort || e?.effort)
}

const reasoningEffortItems = computed(() => {
  const model = selectedModel.value
  const efforts = (model?.reasoning_efforts || []).map(effortValue).filter((v) => !!v)
  const values = efforts.length ? efforts : [...DEFAULT_EFFORTS]
  const current = form.value.reasoning_effort
  if (current && !values.includes(current)) values.push(current)
  const defaultEffort = model?.default_reasoning_effort
  return [
    {
      title: defaultEffort
        ? tm('model.reasoningDefaultWith', { effort: defaultEffort })
        : tm('model.reasoningDefault'),
      value: ''
    },
    ...values.map((v) => ({ title: v, value: v }))
  ]
})

const providerItems = computed(() => {
  const items = [
    { title: tm('model.providerDefault'), value: '' },
    { title: tm('model.providerOpenai'), value: 'openai' }
  ]
  const seen = new Set(['', 'openai'])
  for (const p of form.value.model_providers) {
    const id = p.id.trim()
    if (!id || seen.has(id)) continue
    seen.add(id)
    items.push({ title: p.name.trim() ? `${p.name.trim()} (${id})` : id, value: id })
  }
  const current = form.value.model_provider
  if (current && !seen.has(current)) {
    items.push({ title: tm('model.providerMissing', { id: current }), value: current })
  }
  return items
})

async function loadModels() {
  modelsLoading.value = true
  modelsError.value = ''
  try {
    const res = await httpClient.get('/api/codex/models', { params: { include_hidden: false } })
    if (res.data.status === 'ok' && Array.isArray(res.data.data)) {
      models.value = res.data.data as CodexModel[]
    } else {
      models.value = []
      modelsError.value = res.data.message || tm('messages.modelsLoadFailed')
    }
  } catch (e: any) {
    models.value = []
    modelsError.value = errorMessage(e, tm('messages.modelsLoadFailed'))
  } finally {
    modelsLoading.value = false
  }
}

// ---------- providers ----------

function providerIdRule(v: string): true | string {
  const id = (v || '').trim()
  if (!id) return tm('messages.providerIdRequired')
  if (!PROVIDER_ID_RE.test(id)) return tm('messages.providerIdPattern')
  if (id === 'openai') return tm('messages.providerIdReserved')
  return true
}

function addProvider() {
  form.value.model_providers.push({
    __key: newKey(),
    __showKey: false,
    id: '',
    name: '',
    base_url: '',
    api_key: '',
    wire_api: 'responses'
  })
}

function removeProvider(idx: number) {
  const [removed] = form.value.model_providers.splice(idx, 1)
  if (removed && form.value.model_provider === removed.id.trim()) {
    form.value.model_provider = ''
  }
}

// ---------- config load/save ----------

function normalizeForm(runnerCfg: any): CodexForm {
  const providers = Array.isArray(runnerCfg?.model_providers) ? runnerCfg.model_providers : []
  return {
    model: str(runnerCfg?.model),
    model_provider: str(runnerCfg?.model_provider),
    reasoning_effort: str(runnerCfg?.reasoning_effort),
    model_providers: providers
      .filter((p: any) => p && typeof p === 'object')
      .map((p: any) => ({
        __key: newKey(),
        __showKey: false,
        id: str(p.id),
        name: str(p.name),
        base_url: str(p.base_url),
        api_key: str(p.api_key),
        wire_api: str(p.wire_api) || 'responses'
      })),
    base_instructions: str(runnerCfg?.base_instructions),
    developer_instructions: str(runnerCfg?.developer_instructions)
  }
}

function formPayload(f: CodexForm) {
  return {
    model: (f.model || '').trim(),
    model_provider: f.model_provider || '',
    reasoning_effort: f.reasoning_effort || '',
    model_providers: f.model_providers.map((p) => ({
      id: p.id.trim(),
      name: p.name.trim(),
      base_url: p.base_url.trim(),
      api_key: p.api_key.trim(),
      wire_api: p.wire_api || 'responses'
    })),
    base_instructions: f.base_instructions || '',
    developer_instructions: f.developer_instructions || ''
  }
}

const hasUnsavedChanges = computed(
  () => hasLoaded.value && JSON.stringify(formPayload(form.value)) !== initialSnapshot.value
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
    form.value = normalizeForm(config?.agent_runner?.config)
    initialSnapshot.value = JSON.stringify(formPayload(form.value))
    hasLoaded.value = true
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.loadConfigFailed')), 'error')
  }
}

function validateBeforeSave(): boolean {
  const seen = new Set<string>()
  for (const p of form.value.model_providers) {
    const check = providerIdRule(p.id)
    if (check !== true) {
      toast(check, 'warning')
      return false
    }
    const id = p.id.trim()
    if (seen.has(id)) {
      toast(tm('messages.providerIdDuplicate', { id }), 'warning')
      return false
    }
    seen.add(id)
    if (!p.base_url.trim()) {
      toast(tm('messages.baseUrlRequired'), 'warning')
      return false
    }
  }
  const provider = form.value.model_provider
  if (provider && provider !== 'openai' && !seen.has(provider)) {
    toast(tm('messages.providerNotFound', { id: provider }), 'warning')
    return false
  }
  return true
}

async function save() {
  if (!validateBeforeSave()) return
  saving.value = true
  try {
    // Re-fetch so that only the Codex fields are replaced and every other
    // config field keeps its latest stored value.
    const config = await fetchDefaultConfig()
    if (!config.agent_runner || typeof config.agent_runner !== 'object') {
      config.agent_runner = {}
    }
    if (!config.agent_runner.config || typeof config.agent_runner.config !== 'object') {
      config.agent_runner.config = {}
    }
    const payload = formPayload(form.value)
    Object.assign(config.agent_runner.config, payload)

    const res = await configProfileApi.update(CONFIG_ID, config)
    if (res.data.status === 'ok') {
      initialSnapshot.value = JSON.stringify(payload)
      toast(res.data.message || tm('messages.saveSuccess'))
      loadAccount()
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
    await Promise.all([loadConfig(), loadAccount(), loadModels()])
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
  stopDevicePolling()
})

onBeforeRouteLeave(async () => {
  if (!hasUnsavedChanges.value) return true
  return askForConfirmation(tm('messages.unsavedChangesLeaveConfirm'), confirmDialog)
})
</script>

<style scoped>
@import '@/styles/dashboard-shell.css';

.codex-page {
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

.setting-card {
  border: 1px solid var(--dashboard-border);
  border-radius: 14px;
  padding: 18px;
  background: rgba(var(--v-theme-primary), 0.02);
  min-width: 0;
}

.setting-title {
  font-size: 15px;
  font-weight: 600;
  line-height: 1.5;
}

.setting-subtitle {
  margin-top: 6px;
  color: var(--dashboard-muted);
  font-size: 13px;
  line-height: 1.6;
}

.account-status {
  display: flex;
  align-items: center;
  gap: 14px;
}

.account-status-main {
  flex: 1;
  min-width: 0;
}

.account-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  color: var(--dashboard-muted);
  font-size: 13px;
}

.text-mono,
.device-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.login-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.device-box {
  display: grid;
  gap: 10px;
  padding: 14px;
  border: 1px dashed var(--dashboard-border-strong);
  border-radius: 12px;
}

.device-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  word-break: break-all;
}

.device-label {
  color: var(--dashboard-muted);
  font-size: 13px;
  min-width: 80px;
}

.device-code {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 2px;
}

.device-status {
  font-size: 13px;
  color: var(--dashboard-muted);
}

.provider-list {
  display: grid;
  gap: 14px;
}

.provider-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 2fr) minmax(0, 1.5fr) 140px auto;
  gap: 10px;
  align-items: start;
}

.provider-remove {
  margin-top: 6px;
}

@media (max-width: 1280px) {
  .provider-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    padding-bottom: 14px;
    border-bottom: 1px solid var(--dashboard-border);
  }
}

@media (max-width: 900px) {
  .login-grid,
  .provider-row {
    grid-template-columns: 1fr;
  }

  .account-status {
    flex-wrap: wrap;
  }
}
</style>
