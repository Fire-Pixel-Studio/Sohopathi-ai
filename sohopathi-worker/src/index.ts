import chunks from './chunks.json';

interface Env {
  OLLAMA_API_KEY: string;
  HF_API_KEY: string;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function getQueryEmbedding(text: string, hfApiKey: string) {
  const res = await fetch(
    'https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/pipeline/feature-extraction',
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${hfApiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ inputs: text })
    }
  );
  return res.json() as Promise<number[]>;
}

function findTopChunks(queryEmbedding: number[], topN = 2) {
  const scored = (chunks as any[]).map(chunk => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding)
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN).map(s => s.chunk);
}

async function askModel(question: string, studentClass: string, context: string, ollamaApiKey: string) {
  const prompt = `তুমি "সহপাঠী AI" — বাংলাদেশের ${studentClass || 'ষষ্ঠ থেকে দশম শ্রেণির'} শিক্ষার্থীদের জন্য একজন সহায়ক শিক্ষক। নিচের পাঠ্যবইয়ের অংশ ব্যবহার করে প্রশ্নের উত্তর দাও। উত্তর অবশ্যই বাংলায়, সহজ ভাষায়, ধাপে ধাপে দাও।

পাঠ্যবই অংশ:
${context}

প্রশ্ন: ${question}

উত্তর:`;

  const res = await fetch('https://ollama.com/api/chat', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${ollamaApiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'gpt-oss:120b-cloud',
      messages: [{ role: 'user', content: prompt }],
      stream: false
    })
  });
  const data: any = await res.json();
  return data.message?.content || 'দুঃখিত, উত্তর তৈরি করা যায়নি।';
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Use POST' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }

    try {
      const { question, studentClass } = await request.json() as { question: string; studentClass: string };

      if (!question || question.trim() === '') {
        return new Response(JSON.stringify({ error: 'প্রশ্ন খালি রাখা যাবে না' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      const queryEmbedding = await getQueryEmbedding(question, env.HF_API_KEY);
      const topChunks = findTopChunks(queryEmbedding, 2);
      const context = topChunks.map((c: any) => c.text).join('\n\n');

      const answer = await askModel(question, studentClass, context, env.OLLAMA_API_KEY);

      return new Response(JSON.stringify({ answer }), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });

    } catch (err: any) {
      return new Response(JSON.stringify({ error: err.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }
  }
};