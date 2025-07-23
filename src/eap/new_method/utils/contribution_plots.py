import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interactive_output, HBox, VBox, Output, HTML
from IPython.display import display
import torch
import torch.nn.functional as F
import numpy as np
from utils.metrics import compare_token_logit, compare_token_probability, kl_divergence

def plot_probability_distribution_plotly(logits, title, model, top_n=10):
	"""Plots the probability distribution for the top N tokens using Plotly."""
	probs = F.softmax(logits.to(torch.float32), dim=-1).cpu().numpy()
	idxs = np.argsort(probs)[-top_n:][::-1]
	probs_top = probs[idxs]
	tokens_top = [model.to_string([int(i)]) for i in idxs]

	fig = go.Figure(go.Bar(
		y=tokens_top,
		x=probs_top,
		orientation='h',
	))
	fig.update_layout(
		title=title,
		xaxis_title="Probability",
		yaxis_title="Token",
		yaxis=dict(autorange="reversed"),
		xaxis=dict(range=[0,1]),
		height=300, width=600,
		margin=dict(l=100, r=20, t=50, b=40),
		title_font_size=16,
		xaxis_title_font_size=12,
		yaxis_title_font_size=12,
	)
	return fig

def create_interactive_contribution_plot(
	model,
	cache,
	logits,
	complete_paths,
	all_path_messages,
	random_path_messages,
	prompts_fixed_pos,
	answers_fixed_pos,
	target_tokens_int,
):
	# --- Build controls once ---
	style = {'description_width': '120px'}
	layout = widgets.Layout(width='600px', margin='10px 20px')

	k_slider = widgets.IntSlider(
		value=min(25, len(complete_paths)),
		min=0, max=len(complete_paths), step=1,
		description="Top k paths:", style=style, layout=layout,
		continuous_update=False  # Prevent intermediate updates
	)
	idx_slider = widgets.IntSlider(
		value=0,
		min=0, max=len(prompts_fixed_pos)-1, step=1,
		description="Prompt Index:", style=style, layout=layout,
		continuous_update=False  # Prevent intermediate updates
	)

	# Create persistent Output widgets for plots
	o1, o2, o3 = Output(), Output(), Output()
	fig_box = HBox([o1, o2, o3])
	header_html = HTML()

	def update_plot(k, i):
		# --- Compute residuals & logits ---
		key = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
		orig_resid = cache[key][i:i+1, -1:, :].to(torch.float32)
		if k > 0:
			topk = torch.sum(torch.stack(all_path_messages[:k]), dim=0)[i:i+1,-1:,:].to(torch.float32)
			randk = (
				torch.sum(torch.stack(random_path_messages[:k]), dim=0)[i:i+1,-1:,:].to(torch.float32)
			)
		else:
			topk = torch.zeros_like(orig_resid)
			randk = torch.zeros_like(orig_resid)

		orig_logits = logits[i,-1,:].to(torch.float32)
		corr_logits = model.unembed(model.ln_final(orig_resid - topk)).squeeze()
		rand_logits = model.unembed(model.ln_final(orig_resid - randk)).squeeze()

		# --- Metrics & header ---
		tgt = target_tokens_int[i]
		p_txt = prompts_fixed_pos[i]
		a_txt = answers_fixed_pos[i][0]
		ld = -compare_token_logit(orig_resid, orig_resid-topk, model, [tgt])
		pd = -compare_token_probability(orig_resid, orig_resid-topk, model, [tgt])
		kd = kl_divergence(orig_resid, orig_resid-topk, model, [tgt])

		header_html.value = f"""
		<div style="font-family: 'Segoe UI', Tahoma, sans-serif; margin:4px">
		 <div style="background:#f4f6f9; padding:4px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1)">
		  <div style="display:flex; justify-content:space-between; gap:10px">
			<div style="flex:1.5">
			  <div style="font-weight:600; color:#34495e; margin-bottom:5px">
				Prompt:
				<span style="font-family:monospace; padding:3px; border-radius:4px; color:#2c3e50">
				  {p_txt}
				</span>
			  </div>
			  <div style="font-weight:600; color:#34495e; margin-bottom:5px">
				Correct Token:
				<span style="font-family:monospace; background:#fff; padding:2px 4px; border-radius:4px; color:#2c3e50">
				  {a_txt}
				</span>
			  </div>
			</div>
			<div style="flex:1">
			  <h4 style="color:#34495e; margin:0 0 10px; font-size:1em">
				Effect of removing the top {k} paths on '{model.to_string(tgt)}'
			  </h4>
			  <div style="text-align:center; font-size:1.2em; color:#2c3e50">
				Logit Diff: <b>{ld:.2f}%</b> – 
				Prob Diff: <b>{pd:.2f}%</b> – 
				KL Div: <b>{kd:.4f}</b>
			  </div>
			</div>
		  </div>
		 </div>
		</div>
		"""

		# --- Clear previous plots and display new ones ---
		with o1: o1.clear_output(wait=True)
		with o2: o2.clear_output(wait=True)
		with o3: o3.clear_output(wait=True)
			
		f1 = plot_probability_distribution_plotly(orig_logits, "Original", model)
		f2 = plot_probability_distribution_plotly(corr_logits, f"Top {k} Removed", model)
		f3 = plot_probability_distribution_plotly(rand_logits, f"{k} Random Removed", model)
		
		with o1: display(f1)
		with o2: display(f2)
		with o3: display(f3)

	# Link controls to update function
	control = interactive_output(update_plot, {'k': k_slider, 'i': idx_slider})
	
	# Display the entire layout
	display(VBox([
		header_html,
		fig_box,
		HBox([k_slider, idx_slider]),
		control
	], layout=widgets.Layout(margin="20px")))