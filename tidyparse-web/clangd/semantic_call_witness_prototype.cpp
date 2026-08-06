// This file is a compile-checked prototype, not part of clangd's build.
//
// It records the exact Clang 21 APIs needed by CodeComplete.cpp to validate a
// correlated member-call argument sequence. Production code should keep the
// same whole-sequence contract, add budgets/deduplication, and serialize the
// selected specialization and expression profiles rather than an Expr pointer.

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang::clangd {
namespace {

enum class PrototypeArgumentKind {
  Opaque,
  IntegerZero,
  FloatingZero,
  CharacterZero,
  EmptyString,
  BooleanTrue,
  Nullptr,
};

struct PrototypeArgumentProfile {
  PrototypeArgumentKind Kind = PrototypeArgumentKind::Opaque;
  QualType Type;
  ExprValueKind ValueKind = VK_PRValue;
  ExprObjectKind ObjectKind = OK_Ordinary;
};

struct PrototypeCallWitness {
  FunctionDecl *Specialization = nullptr;
  QualType ResultType;
  ExprValueKind ResultValueKind = VK_PRValue;
  bool Valid = false;
};

Expr *buildPrototypeArgument(ASTContext &AST, SourceLocation Loc,
                             const PrototypeArgumentProfile &Profile) {
  switch (Profile.Kind) {
  case PrototypeArgumentKind::Opaque:
    if (Profile.Type.isNull())
      return nullptr;
    return new (AST) OpaqueValueExpr(Loc, Profile.Type.getNonReferenceType(),
                                     Profile.ValueKind, Profile.ObjectKind);
  case PrototypeArgumentKind::IntegerZero:
    return IntegerLiteral::Create(
        AST, llvm::APInt(AST.getIntWidth(AST.IntTy), 0), AST.IntTy, Loc);
  case PrototypeArgumentKind::FloatingZero:
    return FloatingLiteral::Create(AST, llvm::APFloat(0.0), true, AST.DoubleTy,
                                   Loc);
  case PrototypeArgumentKind::CharacterZero:
    return new (AST)
        CharacterLiteral(0, CharacterLiteralKind::Ascii, AST.CharTy, Loc);
  case PrototypeArgumentKind::EmptyString:
    // The cached literal retains its exact const-char array type and lvalue
    // category. ParenExpr supplies a useful point of instantiation.
    return new (AST)
        ParenExpr(Loc, Loc, AST.getPredefinedStringLiteralFromCache(""));
  case PrototypeArgumentKind::BooleanTrue:
    return CXXBoolLiteralExpr::Create(AST, true, AST.BoolTy, Loc);
  case PrototypeArgumentKind::Nullptr:
    return new (AST) CXXNullPtrLiteralExpr(AST.NullPtrTy, Loc);
  }
  llvm_unreachable("unknown prototype argument kind");
}

/**
 * Performs real member lookup and overload resolution, then recursively
 * instantiates the selected function-template definition. Diagnostic counts
 * still reach DiagnosticErrorTrap while no speculative diagnostic escapes to
 * clangd's client.
 *
 * This mutates the ephemeral completion AST by adding/caching a specialization.
 * Call it only from a disposable code-completion Sema, never a shared preamble
 * AST, and memoize by owner/name/receiver/argument profile.
 */
PrototypeCallWitness probePrototypeMemberCall(
    Sema &S, QualType OwnerType, ExprValueKind ReceiverValueKind,
    llvm::StringRef MemberName, FunctionTemplateDecl &ExpectedPrimary,
    llvm::ArrayRef<PrototypeArgumentProfile> Profiles) {
  PrototypeCallWitness Result;
  ASTContext &AST = S.getASTContext();
  SourceLocation Loc = S.getPreprocessor().getCodeCompletionLoc();
  if (OwnerType.isNull())
    return Result;
  OwnerType = OwnerType.getNonReferenceType();
  CXXRecordDecl *Owner = OwnerType->getAsCXXRecordDecl();
  Owner = Owner ? Owner->getDefinition() : nullptr;
  if (!Owner || Loc.isInvalid())
    return Result;

  IdentifierInfo &Identifier = AST.Idents.get(MemberName);
  LookupResult Members(S, DeclarationName(&Identifier), Loc,
                       Sema::LookupMemberName);
  if (!S.LookupQualifiedName(Members, Owner) || Members.empty())
    return Result;

  auto *Receiver = new (AST) OpaqueValueExpr(
      Loc, OwnerType, ReceiverValueKind, OK_Ordinary);
  llvm::SmallVector<Expr *, 8> Arguments;
  for (const PrototypeArgumentProfile &Profile : Profiles) {
    Expr *Argument = buildPrototypeArgument(AST, Loc, Profile);
    if (!Argument)
      return Result;
    Arguments.push_back(Argument);
  }

  DiagnosticsEngine &Diagnostics = S.getDiagnostics();
  DiagnosticErrorTrap Errors(Diagnostics);
  const bool Suppressed = Diagnostics.getSuppressAllDiagnostics();
  Diagnostics.setSuppressAllDiagnostics(true);
  auto RestoreDiagnostics =
      llvm::make_scope_exit([&] { Diagnostics.setSuppressAllDiagnostics(Suppressed); });
  Sema::TentativeAnalysisScope Tentative(S);

  ExprResult Member = S.BuildMemberReferenceExpr(
      Receiver, OwnerType, Loc, /*IsArrow=*/false, CXXScopeSpec(),
      SourceLocation(), /*FirstQualifierInScope=*/nullptr, Members,
      /*TemplateArgs=*/nullptr, S.getCurScope());
  if (Member.isInvalid() || Errors.hasErrorOccurred())
    return Result;

  ExprResult Call = S.BuildCallExpr(
      /*Scope=*/nullptr, Member.get(), Loc, Arguments, Loc,
      /*ExecConfig=*/nullptr, /*IsExecConfig=*/false,
      /*AllowRecovery=*/false);
  if (Call.isInvalid() || Errors.hasErrorOccurred())
    return Result;

  auto *CallNode = dyn_cast<CallExpr>(Call.get()->IgnoreImplicit());
  FunctionDecl *Callee = CallNode ? CallNode->getDirectCallee() : nullptr;
  FunctionTemplateDecl *SelectedPrimary =
      Callee ? Callee->getPrimaryTemplate() : nullptr;
  if (!SelectedPrimary || SelectedPrimary->getCanonicalDecl() !=
                              ExpectedPrimary.getCanonicalDecl())
    return Result;

  // BuildCallExpr creates the concrete specialization and can enqueue its
  // definition. Recursive=true forces nested implementation templates too;
  // this is what distinguishes a surface-viable forwarding pack from a call
  // whose allocator/tuple construction fails inside the library body.
  if (Callee->getTemplateInstantiationPattern())
    S.InstantiateFunctionDefinition(
        Loc, Callee, /*Recursive=*/true, /*DefinitionRequired=*/true,
        /*AtEndOfTU=*/false);
  if (Errors.hasErrorOccurred() || Callee->isInvalidDecl())
    return Result;

  Result.Specialization = Callee;
  Result.ResultType = Call.get()->getType();
  Result.ResultValueKind = Call.get()->getValueKind();
  Result.Valid = true;
  return Result;
}

CXXConstructExpr *findPrototypeConstruction(Stmt *Root) {
  if (!Root)
    return nullptr;
  if (auto *Construction = dyn_cast<CXXConstructExpr>(Root))
    return Construction;
  for (Stmt *Child : Root->children())
    if (auto *Construction = findPrototypeConstruction(Child))
      return Construction;
  return nullptr;
}

/**
 * Constructor counterpart to probePrototypeMemberCall. Parenthesized and list
 * initialization are deliberately separate witness languages: initializer-list
 * preference and narrowing make it unsound to infer one from the other.
 */
PrototypeCallWitness probePrototypeConstruction(
    Sema &S, QualType TargetType, FunctionTemplateDecl &ExpectedPrimary,
    bool ListInitialization,
    llvm::ArrayRef<PrototypeArgumentProfile> Profiles) {
  PrototypeCallWitness Result;
  if (TargetType.isNull())
    return Result;

  ASTContext &AST = S.getASTContext();
  SourceLocation Loc = S.getPreprocessor().getCodeCompletionLoc();
  TargetType = TargetType.getNonReferenceType();
  if (Loc.isInvalid() || TargetType->isDependentType() ||
      TargetType->isInstantiationDependentType())
    return Result;

  llvm::SmallVector<Expr *, 8> Arguments;
  for (const PrototypeArgumentProfile &Profile : Profiles) {
    Expr *Argument = buildPrototypeArgument(AST, Loc, Profile);
    if (!Argument)
      return Result;
    Arguments.push_back(Argument);
  }

  DiagnosticsEngine &Diagnostics = S.getDiagnostics();
  DiagnosticErrorTrap Errors(Diagnostics);
  const bool Suppressed = Diagnostics.getSuppressAllDiagnostics();
  Diagnostics.setSuppressAllDiagnostics(true);
  auto RestoreDiagnostics = llvm::make_scope_exit(
      [&] { Diagnostics.setSuppressAllDiagnostics(Suppressed); });
  Sema::TentativeAnalysisScope Tentative(S);

  TypeSourceInfo *Type = AST.getTrivialTypeSourceInfo(TargetType, Loc);
  ExprResult Construction = S.BuildCXXTypeConstructExpr(
      Type, Loc, Arguments, Loc, ListInitialization);
  if (Construction.isInvalid() || Errors.hasErrorOccurred())
    return Result;

  CXXConstructExpr *ConstructionNode =
      findPrototypeConstruction(Construction.get());
  FunctionDecl *Callee =
      ConstructionNode ? ConstructionNode->getConstructor() : nullptr;
  FunctionTemplateDecl *SelectedPrimary =
      Callee ? Callee->getPrimaryTemplate() : nullptr;
  if (!SelectedPrimary || SelectedPrimary->getCanonicalDecl() !=
                              ExpectedPrimary.getCanonicalDecl())
    return Result;

  if (Callee->getTemplateInstantiationPattern())
    S.InstantiateFunctionDefinition(
        Loc, Callee, /*Recursive=*/true, /*DefinitionRequired=*/true,
        /*AtEndOfTU=*/false);
  if (Errors.hasErrorOccurred() || Callee->isInvalidDecl())
    return Result;

  Result.Specialization = Callee;
  Result.ResultType = Construction.get()->getType();
  Result.ResultValueKind = Construction.get()->getValueKind();
  Result.Valid = true;
  return Result;
}

} // namespace
} // namespace clang::clangd
