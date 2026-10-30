class InstitutionalRiskGuards:
    def __init__(self, max_slippage_pct=0.015, max_drawdown_pct=0.05, fee_buffer_pct=0.005):
        self.max_slippage_pct = max_slippage_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.fee_buffer_pct = fee_buffer_pct

    def validate_all_or_nothing_basket(self, orders_to_execute, available_liquid_cash):
        """Validates whether liquid cash is sufficient for the total basket order including fee buffers."""
        total_required_ngn = sum(order["amount_ngn"] for order in orders_to_execute if order["action"] == "BUY")
        total_required_with_buffer = total_required_ngn * (1.0 + self.fee_buffer_pct)

        if available_liquid_cash < total_required_with_buffer:
            return {
                "status": "ABORTED",
                "reason": f"Insufficient liquid cash (Available: ₦{available_liquid_cash:,.2f}, Required with fee buffer: ₦{total_required_with_buffer:,.2f})",
                "required": total_required_with_buffer
            }
        return {
            "status": "APPROVED",
            "reason": "Sufficient liquidity verified.",
            "required": total_required_with_buffer
        }

    def calculate_net_allocation_with_fees(self, amount_ngn: float) -> float:
        """Appils fee buffer deductions to trade sizing."""
        return amount_ngn * (1.0 - self.fee_buffer_pct)
